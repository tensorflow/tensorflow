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
#include <fstream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/object_detection_stage.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

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
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";

std::string GetNameFromPath(const std::string& str) {
  int pos = str.find_last_of("/\\");
  if (pos == std::string::npos) return "";
  return str.substr(pos + 1);
}

bool EvaluateModel(const std::string& model_file_path,
                   const std::vector<std::string>& model_labels,
                   const std::vector<std::string>& image_paths,
                   const std::string& ground_truth_proto_file,
                   std::string delegate, std::string output_file_path,
                   int num_interpreter_threads, bool debug_mode) {
  EvaluationStageConfig eval_config;
  eval_config.set_name("object_detection");
  auto* detection_params =
      eval_config.mutable_specification()->mutable_object_detection_params();
  auto* inference_params = detection_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path);
  inference_params->set_num_threads(num_interpreter_threads);
  if (delegate == kNnapiDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::NNAPI);
  } else if (delegate == kGpuDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::GPU);
  }

  // Get ground truth data.
  absl::flat_hash_map<std::string, ObjectDetectionResult> ground_truth_map;
  if (!ground_truth_proto_file.empty()) {
    PopulateGroundTruth(ground_truth_proto_file, &ground_truth_map);
  }

  ObjectDetectionStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init() != kTfLiteOk) return false;

  // Open output file for writing.
  std::ofstream ofile;
  ofile.open(output_file_path, std::ios::out);

  const int step = image_paths.size() / 100;
  for (int i = 0; i < image_paths.size(); ++i) {
    if (step > 1 && i % step == 0) {
      LOG(INFO) << "Finished: " << i / step << "%";
    }

    const std::string image_name = GetNameFromPath(image_paths[i]);
    eval.SetInputs(image_paths[i], ground_truth_map[image_name]);
    if (eval.Run() != kTfLiteOk) return false;

    if (debug_mode) {
      ObjectDetectionResult prediction = *eval.GetLatestPrediction();
      prediction.set_image_name(image_name);
      ofile << prediction.DebugString();
      ofile << "======================================================\n";
    }
  }

  // Write metrics to file.
  EvaluationStageMetrics metrics = eval.LatestMetrics();
  if (ground_truth_proto_file.empty()) {
    // mAP metrics are meaningless for no ground truth.
    metrics.mutable_process_metrics()
        ->mutable_object_detection_metrics()
        ->clear_average_precision_metrics();
  }
  ofile << metrics.DebugString();
  ofile.close();

  return true;
}

int Main(int argc, char* argv[]) {
  // Command Line Flags.
  std::string model_file_path;
  std::string ground_truth_images_path;
  std::string ground_truth_proto_file;
  std::string model_output_labels_path;
  std::string output_file_path;
  std::string delegate;
  int num_interpreter_threads = 1;
  bool debug_mode;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path,
          "Path to labels that correspond to output of model."
          " E.g. in case of COCO-trained SSD model, this is the path to file "
          "where each line contains a class detected by the model in correct "
          "order, starting from background."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filenames"),
      tflite::Flag::CreateFlag(
          kGroundTruthProtoFileFlag, &ground_truth_proto_file,
          "Path to file containing "
          "tflite::evaluation::ObjectDetectionGroundTruth "
          "proto in text format. If left empty, mAP numbers are not output."),
      tflite::Flag::CreateFlag(
          kOutputFilePathFlag, &output_file_path,
          "File to output to. Contains only metrics proto if debug_mode is "
          "off, and per-image predictions also otherwise."),
      tflite::Flag::CreateFlag(kDebugModeFlag, &debug_mode,
                               "Whether to enable debug mode. Per-image "
                               "predictions are written to the output file "
                               "along with metrics."),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(kDelegateFlag, &delegate,
                               "Delegate to use for inference, if available. "
                               "Must be one of {'nnapi', 'gpu'}"),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  // Process images in filename-sorted order.
  std::vector<std::string> image_paths;
  TF_LITE_ENSURE_STATUS(GetSortedFileNames(
      StripTrailingSlashes(ground_truth_images_path), &image_paths));

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path, &model_labels)) {
    LOG(ERROR) << "Could not read model output labels file";
    return 0;
  }

  if (!EvaluateModel(model_file_path, model_labels, image_paths,
                     ground_truth_proto_file, delegate, output_file_path,
                     num_interpreter_threads, debug_mode)) {
    LOG(ERROR) << "Could not evaluate model";
    return 0;
  }

  return 0;
}

}  // namespace evaluation
}  // namespace tflite

int main(int argc, char* argv[]) {
  return tflite::evaluation::Main(argc, argv);
}
