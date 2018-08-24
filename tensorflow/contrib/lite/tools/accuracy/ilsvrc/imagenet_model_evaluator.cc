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
#include "tensorflow/core/platform/init_main.h"
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

// File pattern for imagenet files.
const char* const kImagenetFilePattern = "*.[jJ][pP][eE][gG]";

}  // namespace

namespace tensorflow {
namespace metrics {

/*static*/ Status ImagenetModelEvaluator::Create(
    int argc, char* argv[],
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
      tensorflow::Flag("model_file", &params.model_file_path,
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

  if (params.number_of_images < 0) {
    return errors::InvalidArgument("Invalid: num_examples");
  }

  utils::ModelInfo model_info;
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      utils::GetTFliteModelInfo(params.model_file_path, &model_info),
      "Invalid TFLite model.");

  *model_evaluator =
      absl::make_unique<ImagenetModelEvaluator>(model_info, params);
  return Status::OK();
}

Status ImagenetModelEvaluator::EvaluateModel() {
  if (model_info_.input_shapes.size() != 1) {
    return errors::InvalidArgument("Invalid input shape");
  }

  const TensorShape& input_shape = model_info_.input_shapes[0];
  // Input should be of the shape {1, height, width, 3}
  if (input_shape.dims() != 4 || input_shape.dim_size(3) != 3) {
    return errors::InvalidArgument("Invalid input shape for the model.");
  }

  const int image_height = input_shape.dim_size(1);
  const int image_width = input_shape.dim_size(2);
  const bool is_quantized = (model_info_.input_types[0] == DT_UINT8);

  RunTFLiteModelStage::Params tfl_model_params;
  tfl_model_params.model_file_path = params_.model_file_path;
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
  std::vector<string> model_labels;
  TF_RETURN_IF_ERROR(
      utils::ReadFileLines(params_.model_output_labels_path, &model_labels));
  if (model_labels.size() != 1001) {
    return errors::InvalidArgument("Invalid number of labels: ",
                                   model_labels.size());
  }

  ImagenetTopKAccuracy eval(model_labels, params_.num_ranks);
  std::unique_ptr<EvalPipeline> eval_pipeline;

  auto build_status = builder.WithInputStage(&reader)
                          .WithPreprocessingStage(&inc)
                          .WithRunModelStage(&tfl_model_stage)
                          .WithAccuracyEval(&eval)
                          .WithInput("input_file", DT_STRING)
                          .Build(root, &eval_pipeline);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(build_status,
                                  "Failure while building eval pipeline.");

  std::unique_ptr<Session> session(NewSession(SessionOptions()));

  TF_RETURN_IF_ERROR(eval_pipeline->AttachSession(std::move(session)));
  string data_path =
      StripTrailingSlashes(params_.ground_truth_images_path) + "/";

  const string imagenet_file_pattern = data_path + kImagenetFilePattern;
  std::vector<string> image_files;
  TF_CHECK_OK(
      Env::Default()->GetMatchingPaths(imagenet_file_pattern, &image_files));
  std::vector<string> image_labels;
  TF_CHECK_OK(
      utils::ReadFileLines(params_.ground_truth_labels_path, &image_labels));
  CHECK_EQ(image_files.size(), image_labels.size());

  // Process files in filename sorted order.
  std::sort(image_files.begin(), image_files.end());
  if (params_.number_of_images > 0) {
    image_files = GetFirstN(image_files, params_.number_of_images);
    image_labels = GetFirstN(image_labels, params_.number_of_images);
  }

  for (Observer* observer : observers_) {
    observer->OnEvaluationStart(image_files.size());
  }

  for (int i = 0; i < image_files.size(); i++) {
    TF_CHECK_OK(eval_pipeline->Run(CreateStringTensor(image_files[i]),
                                   CreateStringTensor(image_labels[i])));
    auto stats = eval.GetTopKAccuracySoFar();

    for (Observer* observer : observers_) {
      observer->OnSingleImageEvaluationComplete(stats, image_files[i]);
    }
  }
  return Status::OK();
}

}  // namespace metrics
}  // namespace tensorflow
