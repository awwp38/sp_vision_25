#ifndef AUTO_AIM__TRT_INFER_HPP
#define AUTO_AIM__TRT_INFER_HPP

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace auto_aim
{

class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      fmt::print("[TRT {}] {}\n", severity, msg);
    }
  }
};

struct TRTEngine
{
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::vector<void *> buffers;    // GPU buffers [input, output, ...]
  std::vector<size_t> buffer_sizes;
  cudaStream_t stream = nullptr;

  nvinfer1::Dims input_dims;
  nvinfer1::Dims output_dims;
  int input_index;
  int output_index;
  size_t input_size;
  size_t output_size;

  ~TRTEngine()
  {
    for (auto & buf : buffers) {
      if (buf) cudaFree(buf);
    }
    if (stream) cudaStreamDestroy(stream);
  }
};

inline std::unique_ptr<TRTEngine> load_engine(const std::string & engine_path)
{
  std::ifstream file(engine_path, std::ios::binary);
  if (!file.good()) {
    throw std::runtime_error("Failed to open engine file: " + engine_path);
  }

  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> data(size);
  file.read(data.data(), size);
  file.close();

  Logger logger;
  auto trt = std::make_unique<TRTEngine>();
  trt->runtime.reset(nvinfer1::createInferRuntime(logger));
  trt->engine.reset(trt->runtime->deserializeCudaEngine(data.data(), size));

  if (!trt->engine) {
    throw std::runtime_error("Failed to deserialize TensorRT engine");
  }

  trt->context.reset(trt->engine->createExecutionContext());
  trt->input_index = trt->engine->getBindingIndex("images");
  trt->output_index = trt->engine->getBindingIndex("output0");
  trt->input_dims = trt->engine->getBindingDimensions(trt->input_index);
  trt->output_dims = trt->engine->getBindingDimensions(trt->output_index);

  // Set dynamic batch = 1
  trt->context->setBindingDimensions(trt->input_index, trt->input_dims);

  trt->input_size = 1;
  for (int i = 0; i < trt->input_dims.nbDims; ++i) {
    trt->input_size *= trt->input_dims.d[i];
  }
  trt->input_size *= sizeof(float);

  trt->output_size = 1;
  for (int i = 0; i < trt->output_dims.nbDims; ++i) {
    trt->output_size *= trt->output_dims.d[i];
  }
  trt->output_size *= sizeof(float);

  cudaMalloc(&trt->buffers.resize(2, nullptr), 0);  // placeholder
  trt->buffers.clear();
  trt->buffers.push_back(nullptr);
  trt->buffers.push_back(nullptr);
  cudaMalloc(&trt->buffers[trt->input_index], trt->input_size);
  cudaMalloc(&trt->buffers[trt->output_index], trt->output_size);

  cudaStreamCreate(&trt->stream);

  return trt;
}

/**
 * Preprocess BGR image for TensorRT inference:
 *   BGR → RGB, resize to 640×640, normalize /255,
 *   return float blob on CPU (caller cudaMemcpy to GPU)
 */
inline std::vector<float> preprocess_trt(const cv::Mat & bgr_img, int input_w = 640, int input_h = 640)
{
  // Letterbox resize (keep aspect ratio, pad with gray)
  cv::Mat rgb, resized;
  cv::cvtColor(bgr_img, rgb, cv::COLOR_BGR2RGB);

  float scale = std::min(
    static_cast<float>(input_w) / rgb.cols,
    static_cast<float>(input_h) / rgb.rows);
  int new_w = static_cast<int>(rgb.cols * scale);
  int new_h = static_cast<int>(rgb.rows * scale);
  cv::resize(rgb, resized, {new_w, new_h});

  cv::Mat padded(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

  // HWC → CHW + normalize
  padded.convertTo(padded, CV_32FC3, 1.0 / 255.0);

  std::vector<float> blob(input_w * input_h * 3);
  std::vector<cv::Mat> channels(3);
  cv::split(padded, channels);
  for (int c = 0; c < 3; ++c) {
    std::memcpy(blob.data() + c * input_w * input_h, channels[c].data, input_w * input_h * sizeof(float));
  }

  return blob;
}

}  // namespace auto_aim

#endif  // AUTO_AIM__TRT_INFER_HPP