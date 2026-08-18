// C++ inference node: YOLOv8 (ONNX) detection with OpenCV DNN.
//
// Deployment-style counterpart to the Python harness: same model, same
// driving classes, latency measured on the same machine.
//
// Build:  cmake -S cpp -B cpp/build && cmake --build cpp/build
// Run:    ./cpp/build/detect yolov8n.onnx path/to/image.jpg
// Bench:  ./cpp/build/detect yolov8n.onnx path/to/image.jpg --bench 100
//
// Detections are printed to stdout and an annotated copy is written to
// runs/cpp_out/. Bench results append to runs/cpp_bench.csv.

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

constexpr int kInputSize = 640;
constexpr float kConfThreshold = 0.25f;
constexpr float kNmsThreshold = 0.45f;

// COCO ids -> names, restricted to driving-relevant classes (same as src/evaluate.py)
const std::map<int, std::string> kDrivingClasses = {
    {0, "person"},        {1, "bicycle"}, {2, "car"},   {3, "motorcycle"},
    {5, "bus"},           {6, "train"},   {7, "truck"}, {9, "traffic light"},
    {11, "stop sign"},
};

struct Detection {
  cv::Rect box;
  int class_id;
  float confidence;
};

// Letterbox to kInputSize x kInputSize, keeping aspect ratio. Returns the
// scale and padding needed to map boxes back to the original image.
cv::Mat Letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) {
  scale = std::min(kInputSize / static_cast<float>(src.cols),
                   kInputSize / static_cast<float>(src.rows));
  const int w = static_cast<int>(src.cols * scale);
  const int h = static_cast<int>(src.rows * scale);
  pad_x = (kInputSize - w) / 2;
  pad_y = (kInputSize - h) / 2;

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(w, h));
  cv::Mat out(kInputSize, kInputSize, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(out(cv::Rect(pad_x, pad_y, w, h)));
  return out;
}

// YOLOv8 ONNX output is [1, 84, 8400]: 4 box coords + 80 class scores,
// no objectness. Transpose to rows and decode.
std::vector<Detection> Decode(const cv::Mat& output, float scale, int pad_x,
                              int pad_y) {
  const cv::Mat rows = output.reshape(1, output.size[1]).t();  // 8400 x 84

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < rows.rows; ++i) {
    const float* row = rows.ptr<float>(i);
    const cv::Mat class_scores(1, rows.cols - 4, CV_32F,
                               const_cast<float*>(row + 4));
    cv::Point best;
    double best_score;
    cv::minMaxLoc(class_scores, nullptr, &best_score, nullptr, &best);

    if (best_score < kConfThreshold) continue;
    if (kDrivingClasses.find(best.x) == kDrivingClasses.end()) continue;

    const float cx = row[0], cy = row[1], w = row[2], h = row[3];
    boxes.emplace_back(static_cast<int>((cx - w / 2 - pad_x) / scale),
                       static_cast<int>((cy - h / 2 - pad_y) / scale),
                       static_cast<int>(w / scale),
                       static_cast<int>(h / scale));
    scores.push_back(static_cast<float>(best_score));
    class_ids.push_back(best.x);
  }

  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, kConfThreshold, kNmsThreshold, keep);

  std::vector<Detection> detections;
  detections.reserve(keep.size());
  for (int idx : keep) {
    detections.push_back({boxes[idx], class_ids[idx], scores[idx]});
  }
  return detections;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: detect <model.onnx> <image> [--bench N]\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string image_path = argv[2];
  int bench_iters = 0;
  if (argc >= 5 && std::string(argv[3]) == "--bench") {
    bench_iters = std::stoi(argv[4]);
  }

  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "cannot read image: " << image_path << "\n";
    return 1;
  }

  cv::dnn::Net net = cv::dnn::readNetFromONNX(model_path);

  float scale;
  int pad_x, pad_y;
  const cv::Mat padded = Letterbox(image, scale, pad_x, pad_y);
  const cv::Mat blob = cv::dnn::blobFromImage(
      padded, 1.0 / 255.0, cv::Size(kInputSize, kInputSize), cv::Scalar(),
      /*swapRB=*/true, /*crop=*/false);

  auto forward = [&]() {
    net.setInput(blob);
    return net.forward();
  };

  // Single annotated run
  cv::Mat output = forward();
  const auto detections = Decode(output, scale, pad_x, pad_y);
  std::cout << detections.size() << " driving-class detections:\n";
  for (const auto& det : detections) {
    const auto& name = kDrivingClasses.at(det.class_id);
    std::cout << "  " << name << "  conf=" << det.confidence << "  box="
              << det.box << "\n";
    cv::rectangle(image, det.box, cv::Scalar(0, 200, 0), 2);
    cv::putText(image, name, det.box.tl() + cv::Point(0, -5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 200, 0), 2);
  }
  fs::create_directories("runs/cpp_out");
  const std::string out_path =
      "runs/cpp_out/" + fs::path(image_path).filename().string();
  cv::imwrite(out_path, image);
  std::cout << "annotated image -> " << out_path << "\n";

  // Optional latency benchmark
  if (bench_iters > 0) {
    for (int i = 0; i < 10; ++i) forward();  // warmup
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) forward();
    const auto elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    const double ms = elapsed / bench_iters * 1000.0;
    std::cout << "C++ (OpenCV DNN, CPU): " << ms << " ms/frame  ("
              << bench_iters / elapsed << " FPS)\n";

    fs::create_directories("runs");
    const bool is_new = !fs::exists("runs/cpp_bench.csv");
    std::ofstream csv("runs/cpp_bench.csv", std::ios::app);
    if (is_new) csv << "model,backend,ms_per_frame,fps,iters\n";
    csv << model_path << ",opencv-dnn-cpu," << ms << ","
        << bench_iters / elapsed << "," << bench_iters << "\n";
    std::cout << "appended to runs/cpp_bench.csv\n";
  }
  return 0;
}
