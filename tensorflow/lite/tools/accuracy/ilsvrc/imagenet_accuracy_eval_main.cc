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

#include "tensorflow/lite/tools/accuracy/ilsvrc/imagenet_accuracy_eval.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"

namespace {
constexpr char kNumEvalThreadsFlag[] = "num_eval_threads";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kProtoOutputFilePathFlag[] = "proto_output_file_path";
}  // namespace

int main(int argc, char* argv[]) {
  std::string output_file_path, proto_output_file_path;
  int num_eval_threads = 4;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kNumEvalThreadsFlag, &num_eval_threads,
                               "Number of threads used for evaluation."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path,
                               "Path to output file."),
      tflite::Flag::CreateFlag(kProtoOutputFilePathFlag,
                               &proto_output_file_path,
                               "Path to proto output file."),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  if (num_eval_threads <= 0) {
    LOG(ERROR) << "Invalid number of threads.";
    return EXIT_FAILURE;
  }

  tflite::evaluation::DelegateProviders delegate_providers;
  delegate_providers.InitFromCmdlineArgs(&argc, const_cast<const char**>(argv));

  std::unique_ptr<tensorflow::metrics::ImagenetModelEvaluator> evaluator =
      tensorflow::metrics::CreateImagenetModelEvaluator(&argc, argv,
                                                        num_eval_threads);

  if (!evaluator) {
    LOG(ERROR) << "Fail to create the ImagenetModelEvaluator.";
    return EXIT_FAILURE;
  }

  std::unique_ptr<tensorflow::metrics::ResultsWriter> writer =
      tensorflow::metrics::CreateImagenetEvalResultsWriter(
          evaluator->params().num_ranks, output_file_path);
  if (!writer) {
    LOG(ERROR) << "Fail to create the ResultsWriter.";
    return EXIT_FAILURE;
  }

  evaluator->AddObserver(writer.get());
  LOG(ERROR) << "Starting evaluation with: " << num_eval_threads << " threads.";
  if (evaluator->EvaluateModel(&delegate_providers) != kTfLiteOk) {
    LOG(ERROR) << "Failed to evaluate the model!";
    return EXIT_FAILURE;
  }

  writer->OutputEvalMetriccProto(proto_output_file_path);
  return EXIT_SUCCESS;
}
