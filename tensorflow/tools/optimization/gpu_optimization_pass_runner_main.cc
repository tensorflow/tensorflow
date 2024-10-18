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
// This file creates a binary that can run any registered optimization pass.
// ./xla_gpu_opt  --input_file_path=/tmp/input.pbtxt
// --output_file_path=/tmp/output.pbtxt
// --optimization_pass=NameOfGraphOptimizationPass

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/tools/optimization/optimization_pass_runner.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace {
absl::Status RealMain(int argc, char** argv) {
  string input_file_path;
  string output_file_path;
  string optimization_pass;

  const std::vector<Flag> flag_list = {
      Flag("input_file_path", &input_file_path, "Location of the input graph."),
      Flag("output_file_path", &output_file_path,
           "Location to write the resulting graph."),
      // For now only a single optimization pass can be run.
      Flag("optimization_pass", &optimization_pass,
           "Which optimization pass to run."),
  };
  if (!Flags::Parse(&argc, argv, flag_list)) {
    return errors::FailedPrecondition("Invalid flags passed");
  }
  port::InitMain(argv[0], &argc, &argv);

  if (input_file_path.empty()) {
    return errors::FailedPrecondition("input_file_path is a required flag.");
  }
  if (output_file_path.empty()) {
    return errors::FailedPrecondition("output_file_path is a required flag.");
  }
  if (optimization_pass.empty()) {
    return errors::FailedPrecondition("optimization_pass is a required flag.");
  }

  GraphDef graphdef_input;
  TF_RETURN_IF_ERROR(
      ReadTextProto(Env::Default(), input_file_path, &graphdef_input));

  tensorflow::OptimizationPassRunner runner;

  // Most machines in our servers currently use 8 gpus. There is nothing special
  // about this number and it can be decreased or increased to test other
  // configurations.
  TF_RETURN_IF_ERROR(runner.AddCpus(8));
  TF_RETURN_IF_ERROR(runner.AddGpus(8));

  // This binary is used to test TF:XLA behavior, so turn on auto_jit.
  TF_RETURN_IF_ERROR(runner.SetJitLevel(tensorflow::OptimizerOptions::ON_2));
  GraphDef graphdef_output;
  TF_RETURN_IF_ERROR(runner.Run(optimization_pass, std::move(graphdef_input),
                                &graphdef_output));
  return WriteTextProto(Env::Default(), output_file_path, graphdef_output);
}
}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  TF_CHECK_OK(tensorflow::RealMain(argc, argv));
  return 0;
}
