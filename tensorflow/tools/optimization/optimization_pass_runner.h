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
#ifndef TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
#define TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// OptimizationPassRunner can be initialized, populated with devices, then run
// to test individual Tensorflow Optimization passes.
class OptimizationPassRunner {
 public:
  explicit OptimizationPassRunner()
      : jit_level_(OptimizerOptions::GlobalJitLevel::
                       OptimizerOptions_GlobalJitLevel_DEFAULT) {}

  // Add a fake device to the (initially empty) DeviceSet used for optimization.
  // Names are of the form: "/job:localhost/replica:0/task:0/device:CPU:0"
  Status AddDevice(const string& name, const string& type);

  // Increasing the Jit level will cause XLA to compile parts of the tensorflow
  // graph that it is able to.
  Status SetJitLevel(OptimizerOptions::GlobalJitLevel jit_level);

  // This can be called after adding devices and setting the jit level to parse
  // command line flags and run the specified job. All 3 flags are required:
  // input_file_path, output_file_path, optimization_pass.
  //
  // If this library becomes heavily used, the caller should be responsible for
  // parsing any command line flags desired rather than this Method handling the
  // work of a main() function.
  Status RunMain(int argc, char** argv);

 private:
  OptimizerOptions::GlobalJitLevel jit_level_;
  std::vector<std::unique_ptr<Device>> devices_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_TOOLS_OPTIMIZATION_OPTIMIZATION_PASS_RUNNER_H_
