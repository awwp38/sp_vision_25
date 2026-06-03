#ifndef AUTO_AIM__TRT_YOLO_HPP
#define AUTO_AIM__TRT_YOLO_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/classifier.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tasks/auto_aim/yolos/trt_infer.hpp"

namespace auto_aim
{

/**
 * TensorRT-based YOLO detector for Jetson Orin NX.
 * Replaces OpenVINO inference with TensorRT while keeping the same
 * Armor detection interface (YOLOBase) and postprocessing pipeline.
 *
 * Usage:
 *   1. Convert ONNX → TensorRT engine on Jetson:
 *      trtexec --onnx=yolov8s_pose.onnx --fp16 --saveEngine=yolov8s.engine
 *   2. Set "yolo_name: trt" and "trt_engine_path: assets/yolov8s.engine" in YAML config
 */
class TRTYolo : public YOLOBase
{
public:
  TRTYolo(const std::string & config_path, bool debug);

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  Classifier classifier_;
  Detector detector_;

  std::string engine_path_;
  std::string save_path_;
  bool debug_, use_roi_;

  const int input_w_ = 640;
  const int input_h_ = 640;
  const int class_num_ = 14;       // YOLO-Pose with 14 classes (armor IDs)
  const int keypoint_num_ = 8;     // 4 keypoints × 2 coords
  const float nms_threshold_ = 0.3f;
  const float score_threshold_ = 0.7f;
  double min_confidence_;

  std::unique_ptr<TRTEngine> trt_;

  cv::Rect roi_;
  cv::Point2f offset_;

  // Same postprocessing as YOLOV8 — parse raw output into Armor list
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;
  ArmorType get_type(const Armor & armor);
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);

  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_YOLO_HPP