/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TESTING_COMPILE_MLIR_H_
#define TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TESTING_COMPILE_MLIR_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace testing {

// Compiles the given MLIR module to XLA HLO.
absl::StatusOr<XlaCompiler::CompilationResult> CompileMlirModule(
    const char* mlir_module_str,
    ConfigProto::Experimental::MlirBridgeRollout rollout_state,
    absl::string_view device_type = "XLA_TPU_JIT");

}  // namespace testing
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TF2XLA_API_V2_TESTING_COMPILE_MLIR_H_
