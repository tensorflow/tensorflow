/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kGroundTruthProtoFileFlag[] = "ground_truth_proto";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDebugModeFlag[] = "debug_mode";
constexpr char kDelegateFlag[] = "delegate";

std::string GetNameFromPath(const std::string& str) {
  int pos = str.find_last_of("/\\");
  if (pos == std::string::npos) return "";
  return str.substr(pos + 1);
}

class CocoObjectDetection : public TaskExecutor {
 public:
  CocoObjectDetection(int* argc, char* argv[]);
  ~CocoObjectDetection() override {}

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> Run() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics) const;
  std::string model_file_path_;
  std::string model_output_labels_path_;
  std::string ground_truth_images_path_;
  std::string ground_truth_proto_file_;
  std::string output_file_path_;
  bool debug_mode_;
  std::string delegate_;
  int num_interpreter_threads_;
  bool allow_fp16_;
  DelegateProviders delegate_providers_;
};

CocoObjectDetection::CocoObjectDetection(int* argc, char* argv[])
    : debug_mode_(false), num_interpreter_threads_(1) {
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path_,
          "Path to labels that correspond to output of model."
          " E.g. in case of COCO-trained SSD model, this is the path to file "
          "where each line contains a class detected by the model in correct "
          "order, starting from background."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path_,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filenames"),
      tflite::Flag::CreateFlag(kGroundTruthProtoFileFlag,
                               &ground_truth_proto_file_,
                               "Path to file containing "
                               "tflite::evaluation::ObjectDetectionGroundTruth "
                               "proto in binary serialized format. If left "
                               "empty, mAP numbers are not output."),
      tflite::Flag::CreateFlag(
          kOutputFilePathFlag, &output_file_path_,
          "File to output to. Contains only metrics proto if debug_mode is "
          "off, and per-image predictions also otherwise."),
      tflite::Flag::CreateFlag(kDebugModeFlag, &debug_mode_,
                               "Whether to enable debug mode. Per-image "
                               "predictions are written to the output file "
                               "along with metrics."),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'xnnpack', 'hexagon'}"),
      tflite::Flag::CreateFlag(
          "allow_fp16", &allow_fp16_,
          "allow fp16"),
  };
  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
  DelegateProviders delegate_providers;
  delegate_providers.InitFromCmdlineArgs(argc, const_cast<const char**>(argv));
}

absl::optional<EvaluationStageMetrics> CocoObjectDetection::Run() {
  // Process images in filename-sorted order.
  std::vector<std::string> image_paths;
  if (GetSortedFileNames(StripTrailingSlashes(ground_truth_images_path_),
                         &image_paths) != kTfLiteOk) {
    return absl::nullopt;
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path_, &model_labels)) {
    TFLITE_LOG(ERROR) << "Could not read model output labels file";
    return absl::nullopt;
  }

  EvaluationStageConfig eval_config;
  eval_config.set_name("object_detection");
  auto* detection_params =
      eval_config.mutable_specification()->mutable_object_detection_params();
  auto* inference_params = detection_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));
  inference_params->set_allow_fp16(allow_fp16_);

  // Get ground truth data.
  absl::flat_hash_map<std::string, ObjectDetectionResult> ground_truth_map;
  if (!ground_truth_proto_file_.empty()) {
    PopulateGroundTruth(ground_truth_proto_file_, &ground_truth_map);
  }

  ObjectDetectionStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  const int step = image_paths.size() / 100;
  for (int i = 0; i < image_paths.size(); ++i) {
    if (step > 1 && i % step == 0) {
      TFLITE_LOG(INFO) << "Finished: " << i / step << "%";
    }

    const std::string image_name = GetNameFromPath(image_paths[i]);
    eval.SetInputs(image_paths[i], ground_truth_map[image_name]);
    if (eval.Run() != kTfLiteOk) return absl::nullopt;

    if (debug_mode_) {
      ObjectDetectionResult prediction = *eval.GetLatestPrediction();
      TFLITE_LOG(INFO) << "Image: " << image_name << "\n";
      for (int i = 0; i < prediction.objects_size(); ++i) {
        const auto& object = prediction.objects(i);
        TFLITE_LOG(INFO) << "Object [" << i << "]";
        TFLITE_LOG(INFO) << "  Score: " << object.score();
        TFLITE_LOG(INFO) << "  Class-ID: " << object.class_id();
        TFLITE_LOG(INFO) << "  Bounding Box:";
        const auto& bounding_box = object.bounding_box();
        TFLITE_LOG(INFO) << "    Normalized Top: "
                         << bounding_box.normalized_top();
        TFLITE_LOG(INFO) << "    Normalized Bottom: "
                         << bounding_box.normalized_bottom();
        TFLITE_LOG(INFO) << "    Normalized Left: "
                         << bounding_box.normalized_left();
        TFLITE_LOG(INFO) << "    Normalized Right: "
                         << bounding_box.normalized_right();
      }
      TFLITE_LOG(INFO)
          << "======================================================\n";
    }
  }

  // Write metrics to file.
  EvaluationStageMetrics latest_metrics = eval.LatestMetrics();
  if (ground_truth_proto_file_.empty()) {
    TFLITE_LOG(WARN) << "mAP metrics are meaningless w/o ground truth.";
    latest_metrics.mutable_process_metrics()
        ->mutable_object_detection_metrics()
        ->clear_average_precision_metrics();
  }

  OutputResult(latest_metrics);
  return absl::make_optional(latest_metrics);
}

void CocoObjectDetection::OutputResult(
    const EvaluationStageMetrics& latest_metrics) const {
  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }
  TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
  const auto object_detection_metrics =
      latest_metrics.process_metrics().object_detection_metrics();
  const auto& preprocessing_latency =
      object_detection_metrics.pre_processing_latency();
  TFLITE_LOG(INFO) << "Preprocessing latency: avg="
                   << preprocessing_latency.avg_us() << "(us), std_dev="
                   << preprocessing_latency.std_deviation_us() << "(us)";
  const auto& inference_latency = object_detection_metrics.inference_latency();
  TFLITE_LOG(INFO) << "Inference latency: avg=" << inference_latency.avg_us()
                   << "(us), std_dev=" << inference_latency.std_deviation_us()
                   << "(us)";
  const auto& precision_metrics =
      object_detection_metrics.average_precision_metrics();
  for (int i = 0; i < precision_metrics.individual_average_precisions_size();
       ++i) {
    const auto ap_metric = precision_metrics.individual_average_precisions(i);
    TFLITE_LOG(INFO) << "Average Precision [IOU Threshold="
                     << ap_metric.iou_threshold()
                     << "]: " << ap_metric.average_precision();
  }
  TFLITE_LOG(INFO) << "Overall mAP: "
                   << precision_metrics.overall_mean_average_precision();
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor(int* argc, char* argv[]) {
  return std::unique_ptr<TaskExecutor>(new CocoObjectDetection(argc, argv));
}

}  // namespace evaluation
}  // namespace tflite
