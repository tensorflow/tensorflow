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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

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
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

bool EvaluateModel(const std::string& model_file_path,
                   const std::vector<ImageLabel>& image_labels,
                   const std::vector<std::string>& model_labels,
                   std::string delegate, std::string output_file_path,
                   int num_interpreter_threads) {
  EvaluationStageConfig eval_config;
  eval_config.set_name("image_classification");
  auto* classification_params = eval_config.mutable_specification()
                                    ->mutable_image_classification_params();
  auto* inference_params = classification_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path);
  inference_params->set_num_threads(num_interpreter_threads);
  if (delegate == kNnapiDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::NNAPI);
  } else if (delegate == kGpuDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::GPU);
  }
  classification_params->mutable_topk_accuracy_eval_params()->set_k(10);

  ImageClassificationStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init() != kTfLiteOk) return false;

  const int step = image_labels.size() / 100;
  for (int i = 0; i < image_labels.size(); ++i) {
    if (step > 1 && i % step == 0) {
      LOG(INFO) << "Evaluated: " << i / step << "%";
    }

    eval.SetInputs(image_labels[i].image, image_labels[i].label);
    if (eval.Run() != kTfLiteOk) return false;
  }

  std::ofstream metrics_ofile;
  metrics_ofile.open(output_file_path, std::ios::out);
  metrics_ofile << eval.LatestMetrics().DebugString();
  metrics_ofile.close();

  return true;
}

int Main(int argc, char* argv[]) {
  // Command Line Flags.
  std::string model_file_path;
  std::string ground_truth_images_path;
  std::string ground_truth_labels_path;
  std::string model_output_labels_path;
  std::string blacklist_file_path;
  std::string output_file_path;
  std::string delegate;
  int num_images = 0;
  int num_interpreter_threads = 1;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path,
          "Path to labels that correspond to output of model."
          " E.g. in case of mobilenet, this is the path to label "
          "file where each label is in the same order as the output"
          " of the model."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"),
      tflite::Flag::CreateFlag(
          kGroundTruthLabelsFlag, &ground_truth_labels_path,
          "Path to ground truth labels, corresponding to alphabetical ordering "
          "of ground truth images."),
      tflite::Flag::CreateFlag(
          kBlacklistFilePathFlag, &blacklist_file_path,
          "Path to blacklist file (optional) where each line is a single "
          "integer that is "
          "equal to index number of blacklisted image."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumImagesFlag, &num_images,
                               "Number of examples to evaluate, pass 0 for all "
                               "examples. Default: 0"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(kDelegateFlag, &delegate,
                               "Delegate to use for inference, if available. "
                               "Must be one of {'nnapi', 'gpu'}"),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  // Process images in filename-sorted order.
  std::vector<std::string> image_files, ground_truth_image_labels;
  TF_LITE_ENSURE_STATUS(GetSortedFileNames(
      StripTrailingSlashes(ground_truth_images_path), &image_files));
  if (!ReadFileLines(ground_truth_labels_path, &ground_truth_image_labels)) {
    LOG(ERROR) << "Could not read ground truth labels file";
    return 0;
  }
  if (image_files.size() != ground_truth_image_labels.size()) {
    LOG(ERROR) << "Number of images and ground truth labels is not same";
    return 0;
  }
  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter out blacklisted/unwanted images.
  TF_LITE_ENSURE_STATUS(
      FilterBlackListedImages(blacklist_file_path, &image_labels));
  if (num_images > 0) {
    image_labels = GetFirstN(image_labels, num_images);
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path, &model_labels)) {
    LOG(ERROR) << "Could not read model output labels file";
    return 0;
  }

  if (!EvaluateModel(model_file_path, image_labels, model_labels, delegate,
                     output_file_path, num_interpreter_threads)) {
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
