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

#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kGroundTruthLabelsFlag[] = "ground_truth_labels";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kBlacklistFilePathFlag[] = "blacklist_file_path";
constexpr char kNumImagesFlag[] = "num_images";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

class ImagenetClassification : public TaskExecutor {
 public:
  ImagenetClassification(int* argc, char* argv[]);
  ~ImagenetClassification() override {}

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> Run() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics) const;
  std::string model_file_path_;
  std::string ground_truth_images_path_;
  std::string ground_truth_labels_path_;
  std::string model_output_labels_path_;
  std::string blacklist_file_path_;
  std::string output_file_path_;
  std::string delegate_;
  int num_images_;
  int num_interpreter_threads_;
  bool allow_fp16_;
  DelegateProviders delegate_providers_;
};

ImagenetClassification::ImagenetClassification(int* argc, char* argv[])
    : num_images_(0), num_interpreter_threads_(1) {
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path_,
          "Path to labels that correspond to output of model."
          " E.g. in case of mobilenet, this is the path to label "
          "file where each label is in the same order as the output"
          " of the model."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path_,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"),
      tflite::Flag::CreateFlag(
          kGroundTruthLabelsFlag, &ground_truth_labels_path_,
          "Path to ground truth labels, corresponding to alphabetical ordering "
          "of ground truth images."),
      tflite::Flag::CreateFlag(
          kBlacklistFilePathFlag, &blacklist_file_path_,
          "Path to blacklist file (optional) where each line is a single "
          "integer that is "
          "equal to index number of blacklisted image."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path_,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumImagesFlag, &num_images_,
                               "Number of examples to evaluate, pass 0 for all "
                               "examples. Default: 0"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'hexagon', 'xnnpack'}"),
      tflite::Flag::CreateFlag(
          "nnapi_allow_fp16", &allow_fp16_,
          "nnapi allow fp16"),
  };
  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);
  delegate_providers_.InitFromCmdlineArgs(argc, const_cast<const char**>(argv));
}

absl::optional<EvaluationStageMetrics> ImagenetClassification::Run() {
  // Process images in filename-sorted order.
  std::vector<std::string> image_files, ground_truth_image_labels;
  if (GetSortedFileNames(StripTrailingSlashes(ground_truth_images_path_),
                         &image_files) != kTfLiteOk) {
    return absl::nullopt;
  }
  if (!ReadFileLines(ground_truth_labels_path_, &ground_truth_image_labels)) {
    TFLITE_LOG(ERROR) << "Could not read ground truth labels file";
    return absl::nullopt;
  }
  if (image_files.size() != ground_truth_image_labels.size()) {
    TFLITE_LOG(ERROR) << "Number of images and ground truth labels is not same";
    return absl::nullopt;
  }
  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter out blacklisted/unwanted images.
  if (FilterBlackListedImages(blacklist_file_path_, &image_labels) !=
      kTfLiteOk) {
    return absl::nullopt;
  }
  if (num_images_ > 0) {
    image_labels = GetFirstN(image_labels, num_images_);
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path_, &model_labels)) {
    TFLITE_LOG(ERROR) << "Could not read model output labels file";
    return absl::nullopt;
  }

  EvaluationStageConfig eval_config;
  eval_config.set_name("image_classification");
  auto* classification_params = eval_config.mutable_specification()
                                    ->mutable_image_classification_params();
  auto* inference_params = classification_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));
  inference_params->set_nnapi_allow_fp16(allow_fp16_);
  classification_params->mutable_topk_accuracy_eval_params()->set_k(10);

  ImageClassificationStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  const int step = image_labels.size() / 100;
  for (int i = 0; i < image_labels.size(); ++i) {
    if (step > 1 && i % step == 0) {
      TFLITE_LOG(INFO) << "Evaluated: " << i / step << "%";
    }
    eval.SetInputs(image_labels[i].image, image_labels[i].label);
    if (eval.Run() != kTfLiteOk) return absl::nullopt;
  }

  const auto latest_metrics = eval.LatestMetrics();
  OutputResult(latest_metrics);
  return absl::make_optional(latest_metrics);
}

void ImagenetClassification::OutputResult(
    const EvaluationStageMetrics& latest_metrics) const {
  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }

  TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
  const auto& metrics =
      latest_metrics.process_metrics().image_classification_metrics();
  const auto& preprocessing_latency = metrics.pre_processing_latency();
  TFLITE_LOG(INFO) << "Preprocessing latency: avg="
                   << preprocessing_latency.avg_us() << "(us), std_dev="
                   << preprocessing_latency.std_deviation_us() << "(us)";
  const auto& inference_latency = metrics.inference_latency();
  TFLITE_LOG(INFO) << "Inference latency: avg=" << inference_latency.avg_us()
                   << "(us), std_dev=" << inference_latency.std_deviation_us()
                   << "(us)";
  const auto& accuracy_metrics = metrics.topk_accuracy_metrics();
  for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
    TFLITE_LOG(INFO) << "Top-" << i + 1
                     << " Accuracy: " << accuracy_metrics.topk_accuracies(i);
  }
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor(int* argc, char* argv[]) {
  return std::unique_ptr<TaskExecutor>(new ImagenetClassification(argc, argv));
}

}  // namespace evaluation
}  // namespace tflite
