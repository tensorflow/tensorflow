/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/inference_profiler_stage.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kNumRunsFlag[] = "num_runs";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";

bool EvaluateModel(const std::string& model_file_path,
                   const std::string& delegate, int num_runs,
                   const std::string& output_file_path,
                   int num_interpreter_threads) {
  // Initialize evaluation stage.
  EvaluationStageConfig eval_config;
  eval_config.set_name("inference_profiling");
  auto* inference_params =
      eval_config.mutable_specification()->mutable_tflite_inference_params();
  inference_params->set_model_file_path(model_file_path);
  inference_params->set_num_threads(num_interpreter_threads);
  // This ensures that latency measurement isn't hampered by the time spent in
  // generating random data.
  inference_params->set_invocations_per_run(3);
  inference_params->set_delegate(ParseStringToDelegateType(delegate));
  if (!delegate.empty() &&
      inference_params->delegate() == TfliteInferenceParams::NONE) {
    LOG(WARNING) << "Unsupported TFLite delegate: " << delegate;
    return false;
  }
  InferenceProfilerStage eval(eval_config);
  if (eval.Init() != kTfLiteOk) return false;

  // Run inference & check diff for specified number of runs.
  for (int i = 0; i < num_runs; ++i) {
    if (eval.Run() != kTfLiteOk) return false;
  }

  // Output latency & diff metrics.
  std::ofstream metrics_ofile;
  metrics_ofile.open(output_file_path, std::ios::out);
  metrics_ofile << eval.LatestMetrics().DebugString();
  metrics_ofile.close();
  return true;
}

int Main(int argc, char* argv[]) {
  // Command Line Flags.
  std::string model_file_path;
  std::string output_file_path;
  std::string delegate;
  int num_runs = 50;
  int num_interpreter_threads = 1;
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumRunsFlag, &num_runs,
                               "Number of runs of test & reference inference "
                               "each. Default value: 50"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads,
          "Number of interpreter threads to use for test inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate,
          "Delegate to use for test inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'hexagon'}"),
  };
  tflite::Flags::Parse(&argc, const_cast<const char**>(argv), flag_list);

  if (!EvaluateModel(model_file_path, delegate, num_runs, output_file_path,
                     num_interpreter_threads)) {
    LOG(ERROR) << "Could not evaluate model!";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

}  // namespace evaluation
}  // namespace tflite

int main(int argc, char* argv[]) {
  return tflite::evaluation::Main(argc, argv);
}
