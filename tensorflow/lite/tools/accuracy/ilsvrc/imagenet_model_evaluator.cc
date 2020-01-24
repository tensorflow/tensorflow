/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"

#include <fstream>
#include <iomanip>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/accuracy/ilsvrc/default/custom_delegates.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace {

using tflite::evaluation::ImageLabel;
using tflite::evaluation::TfliteInferenceParams;

constexpr char kNumImagesFlag[] = "num_images";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kGroundTruthLabelsFlag[] = "ground_truth_labels";
constexpr char kBlacklistFilePathFlag[] = "blacklist_file_path";
constexpr char kModelFileFlag[] = "model_file";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";
constexpr char kNumRanksFlag[] = "num_ranks";

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

template <typename T>
std::vector<std::vector<T>> Split(const std::vector<T>& v, int n) {
  if (n <= 0) {
    return std::vector<std::vector<T>>();
  }
  std::vector<std::vector<T>> vecs(n);
  int input_index = 0;
  int vec_index = 0;
  while (input_index < v.size()) {
    vecs[vec_index].push_back(v[input_index]);
    vec_index = (vec_index + 1) % n;
    input_index++;
  }
  return vecs;
}

}  // namespace

namespace tensorflow {
namespace metrics {

class CompositeObserver : public ImagenetModelEvaluator::Observer {
 public:
  explicit CompositeObserver(const std::vector<Observer*>& observers)
      : observers_(observers) {}

  void OnEvaluationStart(const std::unordered_map<uint64_t, int>&
                             shard_id_image_count_map) override {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto observer : observers_) {
      observer->OnEvaluationStart(shard_id_image_count_map);
    }
  }

  void OnSingleImageEvaluationComplete(
      uint64_t shard_id,
      const tflite::evaluation::TopkAccuracyEvalMetrics& metrics,
      const std::string& image) override {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto observer : observers_) {
      observer->OnSingleImageEvaluationComplete(shard_id, metrics, image);
    }
  }

 private:
  const std::vector<ImagenetModelEvaluator::Observer*>& observers_;
  std::mutex mu_;
};

/*static*/ TfLiteStatus ImagenetModelEvaluator::Create(
    int* argc, char* argv[], int num_threads,
    std::unique_ptr<ImagenetModelEvaluator>* model_evaluator) {
  Params params;
  params.number_of_images = 100;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kNumImagesFlag, &params.number_of_images,
                               "Number of examples to evaluate, pass 0 for all "
                               "examples. Default: 100"),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &params.model_output_labels_path,

          "Path to labels that correspond to output of model."
          " E.g. in case of mobilenet, this is the path to label "
          "file where each label is in the same order as the output"
          " of the model."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &params.ground_truth_images_path,

          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"),
      tflite::Flag::CreateFlag(
          kGroundTruthLabelsFlag, &params.ground_truth_labels_path,
          "Path to ground truth labels, corresponding to alphabetical ordering "
          "of ground truth images."),
      tflite::Flag::CreateFlag(
          kBlacklistFilePathFlag, &params.blacklist_file_path,
          "Path to blacklist file (optional) where each line is a single "
          "integer that is "
          "equal to index number of blacklisted image."),
      tflite::Flag::CreateFlag(kModelFileFlag, &params.model_file_path,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &params.num_interpreter_threads,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(kDelegateFlag, &params.delegate,
                               "Delegate to use for inference, if available. "
                               "Must be one of {'nnapi', 'gpu'}"),
      tflite::Flag::CreateFlag(kNumRanksFlag, &params.num_ranks,
                               "Generates the top-1 to top-k accuracy values"
                               "where k = num_ranks. Default: 10"),
  };
  tflite::Flags::Parse(argc, const_cast<const char**>(argv), flag_list);

  if (params.number_of_images < 0) {
    LOG(ERROR) << "Invalid: num_examples";
    return kTfLiteError;
  }

  *model_evaluator =
      absl::make_unique<ImagenetModelEvaluator>(params, num_threads);
  return kTfLiteOk;
}

TfLiteStatus EvaluateModelForShard(const uint64_t shard_id,
                                   const std::vector<ImageLabel>& image_labels,
                                   const std::vector<std::string>& model_labels,
                                   const ImagenetModelEvaluator::Params& params,
                                   ImagenetModelEvaluator::Observer* observer,
                                   int num_ranks) {
  tflite::evaluation::EvaluationStageConfig eval_config;
  eval_config.set_name("image_classification");
  auto* classification_params = eval_config.mutable_specification()
                                    ->mutable_image_classification_params();
  auto* inference_params = classification_params->mutable_inference_params();
  inference_params->set_model_file_path(params.model_file_path);
  inference_params->set_num_threads(params.num_interpreter_threads);
  if (params.delegate == kNnapiDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::NNAPI);
  } else if (params.delegate == kGpuDelegate) {
    inference_params->set_delegate(TfliteInferenceParams::GPU);
  }
  classification_params->mutable_topk_accuracy_eval_params()->set_k(num_ranks);

  tflite::evaluation::ImageClassificationStage eval(eval_config);
  eval.SetAllLabels(model_labels);
  TF_LITE_ENSURE_STATUS(eval.Init());

  TF_LITE_ENSURE_STATUS(tflite::evaluation::ApplyCustomDelegates(
      params.delegate, params.num_interpreter_threads, &eval));

  for (const auto& image_label : image_labels) {
    eval.SetInputs(image_label.image, image_label.label);

    TF_LITE_ENSURE_STATUS(eval.Run());
    observer->OnSingleImageEvaluationComplete(
        shard_id,
        eval.LatestMetrics()
            .process_metrics()
            .image_classification_metrics()
            .topk_accuracy_metrics(),
        image_label.image);
  }
  return kTfLiteOk;
}

TfLiteStatus ImagenetModelEvaluator::EvaluateModel() const {
  const std::string data_path = tflite::evaluation::StripTrailingSlashes(
                                    params_.ground_truth_images_path) +
                                "/";
  std::vector<std::string> image_files;
  TF_LITE_ENSURE_STATUS(
      tflite::evaluation::GetSortedFileNames(data_path, &image_files));
  std::vector<string> ground_truth_image_labels;
  if (!tflite::evaluation::ReadFileLines(params_.ground_truth_labels_path,
                                         &ground_truth_image_labels))
    return kTfLiteError;
  if (image_files.size() != ground_truth_image_labels.size()) {
    LOG(ERROR) << "Images and ground truth labels don't match";
    return kTfLiteError;
  }

  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter any blacklisted images.
  if (tflite::evaluation::FilterBlackListedImages(params_.blacklist_file_path,
                                                  &image_labels) != kTfLiteOk) {
    LOG(ERROR) << "Could not filter by blacklist";
    return kTfLiteError;
  }

  if (params_.number_of_images > 0) {
    image_labels = GetFirstN(image_labels, params_.number_of_images);
  }

  std::vector<string> model_labels;
  if (!tflite::evaluation::ReadFileLines(params_.model_output_labels_path,
                                         &model_labels)) {
    LOG(ERROR) << "Could not read: " << params_.model_output_labels_path;
    return kTfLiteError;
  }
  if (model_labels.size() != 1001) {
    LOG(ERROR) << "Invalid number of labels: " << model_labels.size();
    return kTfLiteError;
  }

  auto img_labels = Split(image_labels, num_threads_);

  CompositeObserver observer(observers_);

  std::vector<std::thread> thread_pool;
  bool all_okay = true;
  std::unordered_map<uint64_t, int> shard_id_image_count_map;
  thread_pool.reserve(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    const auto& image_label = img_labels[i];
    const uint64_t shard_id = i + 1;
    shard_id_image_count_map[shard_id] = image_label.size();
    auto func = [shard_id, &image_label, &model_labels, this, &observer,
                 &all_okay]() {
      if (EvaluateModelForShard(shard_id, image_label, model_labels, params_,
                                &observer, params_.num_ranks) != kTfLiteOk) {
        all_okay = false;
      }
    };
    thread_pool.push_back(std::thread(func));
  }

  observer.OnEvaluationStart(shard_id_image_count_map);
  for (auto& thread : thread_pool) {
    thread.join();
  }

  return all_okay ? kTfLiteOk : kTfLiteError;
}

}  // namespace metrics
}  // namespace tensorflow
