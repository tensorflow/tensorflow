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

#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_model_evaluator.h"

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/contrib/lite/tools/accuracy/eval_pipeline.h"
#include "tensorflow/contrib/lite/tools/accuracy/eval_pipeline_builder.h"
#include "tensorflow/contrib/lite/tools/accuracy/file_reader_stage.h"
#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_topk_eval.h"
#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/inception_preprocessing.h"
#include "tensorflow/contrib/lite/tools/accuracy/run_tflite_model_stage.h"
#include "tensorflow/contrib/lite/tools/accuracy/utils.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

namespace {
using tensorflow::string;

string StripTrailingSlashes(const string& path) {
  int end = path.size();
  while (end > 0 && path[end - 1] == '/') {
    end--;
  }
  return path.substr(0, end);
}

tensorflow::Tensor CreateStringTensor(const string& value) {
  tensorflow::Tensor tensor(tensorflow::DT_STRING, tensorflow::TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

template <typename T>
std::vector<std::vector<T>> Split(const std::vector<T>& v, int n) {
  CHECK_GT(n, 0);
  std::vector<std::vector<T>> vecs(n);
  int input_index = 0;
  int vec_index = 0;
  while (input_index < v.size()) {
    vecs[vec_index].push_back(v[input_index]);
    vec_index = (vec_index + 1) % n;
    input_index++;
  }
  CHECK_EQ(vecs.size(), n);
  return vecs;
}

// File pattern for imagenet files.
const char* const kImagenetFilePattern = "*.[jJ][pP][eE][gG]";

}  // namespace

namespace tensorflow {
namespace metrics {

class CompositeObserver : public ImagenetModelEvaluator::Observer {
 public:
  explicit CompositeObserver(const std::vector<Observer*>& observers)
      : observers_(observers) {}

  void OnEvaluationStart(const std::unordered_map<uint64_t, int>&
                             shard_id_image_count_map) override {
    mutex_lock lock(mu_);
    for (auto observer : observers_) {
      observer->OnEvaluationStart(shard_id_image_count_map);
    }
  }

  void OnSingleImageEvaluationComplete(
      uint64_t shard_id, const ImagenetTopKAccuracy::AccuracyStats& stats,
      const string& image) override {
    mutex_lock lock(mu_);
    for (auto observer : observers_) {
      observer->OnSingleImageEvaluationComplete(shard_id, stats, image);
    }
  }

 private:
  const std::vector<ImagenetModelEvaluator::Observer*>& observers_
      GUARDED_BY(mu_);
  mutex mu_;
};

/*static*/ Status ImagenetModelEvaluator::Create(
    int argc, char* argv[], int num_threads,
    std::unique_ptr<ImagenetModelEvaluator>* model_evaluator) {
  Params params;
  const std::vector<Flag> flag_list = {
      Flag("model_output_labels", &params.model_output_labels_path,
           "Path to labels that correspond to output of model."
           " E.g. in case of mobilenet, this is the path to label "
           "file where each label is in the same order as the output"
           " of the model."),
      Flag("ground_truth_images_path", &params.ground_truth_images_path,
           "Path to ground truth images."),
      Flag("ground_truth_labels", &params.ground_truth_labels_path,
           "Path to ground truth labels."),
      Flag("num_images", &params.number_of_images,
           "Number of examples to evaluate, pass 0 for all "
           "examples. Default: 100"),
      Flag("blacklist_file_path", &params.blacklist_file_path,
           "Path to blacklist file (optional)."
           "Path to blacklist file where each line is a single integer that is "
           "equal to number of blacklisted image."),
      Flag("model_file", &params.model_file_path,
           "Path to test tflite model file."),
  };
  const bool parse_result = Flags::Parse(&argc, argv, flag_list);
  if (!parse_result)
    return errors::InvalidArgument("Invalid command line flags");
  ::tensorflow::port::InitMain(argv[0], &argc, &argv);

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      Env::Default()->IsDirectory(params.ground_truth_images_path),
      "Invalid ground truth data path.");
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      Env::Default()->FileExists(params.ground_truth_labels_path),
      "Invalid ground truth labels path.");
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      Env::Default()->FileExists(params.model_output_labels_path),
      "Invalid model output labels path.");

  if (!params.blacklist_file_path.empty()) {
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        Env::Default()->FileExists(params.blacklist_file_path),
        "Invalid blacklist path.");
  }

  if (params.number_of_images < 0) {
    return errors::InvalidArgument("Invalid: num_examples");
  }

  utils::ModelInfo model_info;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      utils::GetTFliteModelInfo(params.model_file_path, &model_info),
      "Invalid TFLite model.");

  *model_evaluator = absl::make_unique<ImagenetModelEvaluator>(
      model_info, params, num_threads);
  return Status::OK();
}

struct ImageLabel {
  string image;
  string label;
};

Status EvaluateModelForShard(const uint64_t shard_id,
                             const std::vector<ImageLabel>& image_labels,
                             const std::vector<string>& model_labels,
                             const utils::ModelInfo& model_info,
                             const ImagenetModelEvaluator::Params& params,
                             ImagenetModelEvaluator::Observer* observer,
                             ImagenetTopKAccuracy* eval) {
  const TensorShape& input_shape = model_info.input_shapes[0];
  const int image_height = input_shape.dim_size(1);
  const int image_width = input_shape.dim_size(2);
  const bool is_quantized = (model_info.input_types[0] == DT_UINT8);

  RunTFLiteModelStage::Params tfl_model_params;
  tfl_model_params.model_file_path = params.model_file_path;
  if (is_quantized) {
    tfl_model_params.input_type = {DT_UINT8};
    tfl_model_params.output_type = {DT_UINT8};
  } else {
    tfl_model_params.input_type = {DT_FLOAT};
    tfl_model_params.output_type = {DT_FLOAT};
  }

  Scope root = Scope::NewRootScope();
  FileReaderStage reader;
  InceptionPreprocessingStage inc(image_height, image_width, is_quantized);
  RunTFLiteModelStage tfl_model_stage(tfl_model_params);
  EvalPipelineBuilder builder;

  std::unique_ptr<EvalPipeline> eval_pipeline;

  auto build_status = builder.WithInputStage(&reader)
                          .WithPreprocessingStage(&inc)
                          .WithRunModelStage(&tfl_model_stage)
                          .WithAccuracyEval(eval)
                          .WithInput("input_file", DT_STRING)
                          .Build(root, &eval_pipeline);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(build_status,
                                  "Failure while building eval pipeline.");
  std::unique_ptr<Session> session(NewSession(SessionOptions()));

  TF_RETURN_IF_ERROR(eval_pipeline->AttachSession(std::move(session)));

  for (const auto& image_label : image_labels) {
    TF_CHECK_OK(eval_pipeline->Run(CreateStringTensor(image_label.image),
                                   CreateStringTensor(image_label.label)));
    observer->OnSingleImageEvaluationComplete(
        shard_id, eval->GetTopKAccuracySoFar(), image_label.image);
  }
  return Status::OK();
}

Status FilterBlackListedImages(const string& blacklist_file_path,
                               std::vector<ImageLabel>* image_labels) {
  if (!blacklist_file_path.empty()) {
    std::vector<string> lines;
    TF_RETURN_IF_ERROR(utils::ReadFileLines(blacklist_file_path, &lines));
    std::vector<int> blacklist_ids;
    blacklist_ids.reserve(lines.size());
    // Populate blacklist_ids with indices of images.
    std::transform(lines.begin(), lines.end(),
                   std::back_inserter(blacklist_ids),
                   [](const string& val) { return std::stoi(val) - 1; });

    std::vector<ImageLabel> filtered_images;
    std::sort(blacklist_ids.begin(), blacklist_ids.end());
    const size_t size_post_filtering =
        image_labels->size() - blacklist_ids.size();
    filtered_images.reserve(size_post_filtering);
    int blacklist_index = 0;
    for (int image_index = 0; image_index < image_labels->size();
         image_index++) {
      if (blacklist_index < blacklist_ids.size() &&
          blacklist_ids[blacklist_index] == image_index) {
        blacklist_index++;
        continue;
      }
      filtered_images.push_back((*image_labels)[image_index]);
    }

    if (filtered_images.size() != size_post_filtering) {
      return errors::Internal("Invalid number of filtered images");
    }
    *image_labels = filtered_images;
  }
  return Status::OK();
}

Status ImagenetModelEvaluator::EvaluateModel() const {
  if (model_info_.input_shapes.size() != 1) {
    return errors::InvalidArgument("Invalid input shape");
  }

  const TensorShape& input_shape = model_info_.input_shapes[0];
  // Input should be of the shape {1, height, width, 3}
  if (input_shape.dims() != 4 || input_shape.dim_size(3) != 3) {
    return errors::InvalidArgument("Invalid input shape for the model.");
  }

  string data_path =
      StripTrailingSlashes(params_.ground_truth_images_path) + "/";

  const string imagenet_file_pattern = data_path + kImagenetFilePattern;
  std::vector<string> image_files;
  TF_CHECK_OK(
      Env::Default()->GetMatchingPaths(imagenet_file_pattern, &image_files));
  std::vector<string> ground_truth_image_labels;
  TF_CHECK_OK(utils::ReadFileLines(params_.ground_truth_labels_path,
                                   &ground_truth_image_labels));
  CHECK_EQ(image_files.size(), ground_truth_image_labels.size());

  // Process files in filename sorted order.
  std::sort(image_files.begin(), image_files.end());

  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter any blacklisted images.
  TF_CHECK_OK(
      FilterBlackListedImages(params_.blacklist_file_path, &image_labels));

  if (params_.number_of_images > 0) {
    image_labels = GetFirstN(image_labels, params_.number_of_images);
  }

  std::vector<string> model_labels;
  TF_RETURN_IF_ERROR(
      utils::ReadFileLines(params_.model_output_labels_path, &model_labels));
  if (model_labels.size() != 1001) {
    return errors::InvalidArgument("Invalid number of labels: ",
                                   model_labels.size());
  }

  ImagenetTopKAccuracy eval(model_labels, params_.num_ranks);

  auto img_labels = Split(image_labels, num_threads_);

  BlockingCounter counter(num_threads_);

  CompositeObserver observer(observers_);

  ::tensorflow::thread::ThreadPool pool(Env::Default(), "evaluation_pool",
                                        num_threads_);
  std::unordered_map<uint64_t, int> shard_id_image_count_map;
  std::vector<std::function<void()>> thread_funcs;
  thread_funcs.reserve(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    const auto& image_label = img_labels[i];
    const uint64_t shard_id = i + 1;
    shard_id_image_count_map[shard_id] = image_label.size();
    auto func = [shard_id, &image_label, &model_labels, this, &observer, &eval,
                 &counter]() {
      TF_CHECK_OK(EvaluateModelForShard(shard_id, image_label, model_labels,
                                        model_info_, params_, &observer,
                                        &eval));
      counter.DecrementCount();
    };
    thread_funcs.push_back(func);
  }

  observer.OnEvaluationStart(shard_id_image_count_map);
  for (const auto& func : thread_funcs) {
    pool.Schedule(func);
  }

  counter.Wait();

  return Status::OK();
}

}  // namespace metrics
}  // namespace tensorflow
