#include "trt_yolo.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "tasks/auto_aim/classifier.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{

TRTYolo::TRTYolo(const std::string & config_path, bool debug)
: classifier_(config_path), detector_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  engine_path_ = yaml["trt_engine_path"].as<std::string>();
  min_confidence_ = yaml["min_confidence"].as<double>();

  int x = yaml["roi"]["x"].as<int>();
  int y = yaml["roi"]["y"].as<int>();
  int width = yaml["roi"]["width"].as<int>();
  int height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  tools::logger()->info("Loading TensorRT engine: {}", engine_path_);
  trt_ = load_engine(engine_path_);
  tools::logger()->info("TensorRT engine loaded successfully");
}

std::list<Armor> TRTYolo::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty image, camera drop!");
    return {};
  }

  // ---- ROI crop ----
  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) roi_.width = raw_img.cols;
    if (roi_.height == -1) roi_.height = raw_img.rows;
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // ---- Preprocess (CPU) ----
  auto scale = std::min(
    static_cast<float>(input_w_) / bgr_img.cols,
    static_cast<float>(input_h_) / bgr_img.rows);
  auto new_w = static_cast<int>(bgr_img.cols * scale);
  auto new_h = static_cast<int>(bgr_img.rows * scale);

  // BGR → RGB, resize, pad, normalize → CHW float blob
  auto blob = preprocess_trt(bgr_img, input_w_, input_h_);

  // ---- GPU inference ----
  cudaMemcpyAsync(
    trt_->buffers[trt_->input_index], blob.data(), trt_->input_size,
    cudaMemcpyHostToDevice, trt_->stream);
  trt_->context->enqueueV2(trt_->buffers.data(), trt_->stream, nullptr);

  // ---- Postprocess ----
  // Allocate output buffer on CPU and copy from GPU
  std::vector<float> output_cpu(trt_->output_size / sizeof(float));
  cudaMemcpyAsync(
    output_cpu.data(), trt_->buffers[trt_->output_index], trt_->output_size,
    cudaMemcpyDeviceToHost, trt_->stream);
  cudaStreamSynchronize(trt_->stream);

  // Wrap output as cv::Mat for parse() — shape [1, 56, 8400] for YOLOv8-pose
  // 56 = 4 (bbox) + 14 (classes) + 8 (keypoints) + extra
  auto & out_dims = trt_->output_dims;
  int num_rows = out_dims.d[1];   // number of predictions per scale
  int num_cols = out_dims.d[2];   // 4 + class_num_ + keypoint_num_ features
  cv::Mat output(num_rows, num_cols, CV_32F, output_cpu.data());

  return parse(scale, output, raw_img, frame_count);
}

std::list<Armor> TRTYolo::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  cv::transpose(output, output);  // [N, cols] → [cols, N]

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 4 + class_num_ + keypoint_num_);

    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);
    if (score < score_threshold_) continue;

    std::vector<cv::Point2f> armor_key_points;
    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);

    auto left = static_cast<int>((x - 0.5f * w) / scale);
    auto top = static_cast<int>((y - 0.5f * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    for (int i = 0; i < 4; i++) {
      float kx = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float ky = one_key_points.at<float>(0, i * 2 + 1) / scale;
      armor_key_points.push_back({kx, ky});
    }

    ids.push_back(max_point.x);
    confidences.push_back(static_cast<float>(score));
    boxes.emplace_back(left, top, width, height);
    armors_key_points.push_back(armor_key_points);
  }

  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // Classifier check
  for (auto it = armors.begin(); it != armors.end();) {
    it->pattern = get_pattern(bgr_img, *it);
    classifier_.classify(*it);
    if (!check_name(*it)) { it = armors.erase(it); continue; }
    it->type = get_type(*it);
    if (!check_type(*it)) { it = armors.erase(it); continue; }
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);
  return armors;
}

std::list<Armor> TRTYolo::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

bool TRTYolo::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

bool TRTYolo::check_type(const Armor & armor) const
{
  return (armor.type == ArmorType::small)
           ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
           : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
              armor.name != ArmorName::outpost);
}

ArmorType TRTYolo::get_type(const Armor & armor)
{
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) return ArmorType::big;
  return ArmorType::small;
}

cv::Point2f TRTYolo::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / bgr_img.cols, center.y / bgr_img.rows};
}

cv::Mat TRTYolo::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  auto tl = (armor.points[0] + armor.points[3]) / 2 - (armor.points[3] - armor.points[0]) * 1.125;
  auto bl = (armor.points[0] + armor.points[3]) / 2 + (armor.points[3] - armor.points[0]) * 1.125;
  auto tr = (armor.points[2] + armor.points[1]) / 2 - (armor.points[2] - armor.points[1]) * 1.125;
  auto br = (armor.points[2] + armor.points[1]) / 2 + (armor.points[2] - armor.points[1]) * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);

  if (roi_right <= roi_left || roi_bottom <= roi_top) return {};
  if (roi_right > bgr_img.cols || roi_bottom > bgr_img.rows) return {};
  return bgr_img(cv::Rect(roi_left, roi_top, roi_right - roi_left, roi_bottom - roi_top));
}

void TRTYolo::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  if (keypoints.size() != 4) return;
  std::sort(keypoints.begin(), keypoints.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.y < b.y; });
  std::vector<cv::Point2f> top = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bot = {keypoints[2], keypoints[3]};
  std::sort(top.begin(), top.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });
  std::sort(bot.begin(), bot.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });
  keypoints = {top[0], top[1], bot[1], bot[0]};  // tl, tr, br, bl
}

void TRTYolo::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("TRT detection", detection);
}

}  // namespace auto_aim