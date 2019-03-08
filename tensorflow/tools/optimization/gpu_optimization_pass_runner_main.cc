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

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/tools/optimization/optimization_pass_runner.h"

int main(int argc, char** argv) {
  tensorflow::OptimizationPassRunner runner;
  // Add fake devices for CPU, GPU, and XLA to ensure we have all devices we
  // need.
  // Most machines in our servers currently use 8 gpus. There is nothing special
  // about this number and it can be decreased or increased to test other
  // configurations.
  int num_gpus_per_machine = 8;
  for (int i = 0; i < num_gpus_per_machine; i++) {
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:CPU:", i),
        tensorflow::DEVICE_CPU));
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:GPU:", i),
        tensorflow::DEVICE_GPU));
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:XLA_CPU:", i),
        tensorflow::DEVICE_XLA_CPU));
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:XLA_GPU:", i),
        tensorflow::DEVICE_XLA_GPU));
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:CPU_XLA_JIT:", i),
        tensorflow::DEVICE_CPU_XLA_JIT));
    TF_CHECK_OK(runner.AddDevice(
        absl::StrCat("/job:localhost/replica:0/task:0/device:GPU_XLA_JIT:", i),
        tensorflow::DEVICE_GPU_XLA_JIT));
  }
  // This binary is used to test TF:XLA behavior, so turn on auto_jit.
  TF_CHECK_OK(runner.SetJitLevel(tensorflow::OptimizerOptions::GlobalJitLevel::
                                     OptimizerOptions_GlobalJitLevel_ON_2));
  // Run the actual "main" function.
  TF_CHECK_OK(runner.RunMain(argc, argv));
}
