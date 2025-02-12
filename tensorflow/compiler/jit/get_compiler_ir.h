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
#ifndef TENSORFLOW_COMPILER_JIT_GET_COMPILER_IR_H_
#define TENSORFLOW_COMPILER_JIT_GET_COMPILER_IR_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

class ProcessFunctionLibraryRuntime;
class Device;
class Tensor;
class TensorHandle;
class EagerContext;

enum class IrExportStage {
  STABLEHLO,
  STABLEHLO_SERIALIZED,
  HLO,
  HLO_NO_METADATA,
  HLO_SERIALIZED,
  OPTIMIZED_HLO,
  OPTIMIZED_HLO_SERIALIZED,
  OPTIMIZED_HLO_PROTO_SERIALIZED,
  OPTIMIZED_HLO_DOT
};

struct ArgShapeAndDType {
  TensorShape shape;
  DataType dtype;
};

enum class CompilerArgSource {
  TENSOR_SPEC,
  CONCRETE_INPUT,
};

// Returns the IR format of the selected stage for a given function `func_name`
// using library runtime `runtime` on a device `dev` with given
// `inputs_arg_shape_and_dtype` and `input_handles`.
absl::StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, Device* dev, EagerContext* context,
    absl::Span<const ArgShapeAndDType> input_arg_shape_and_dtype,
    absl::Span<const TensorHandle* const> input_handles,
    CompilerArgSource compiler_arg_source);

// Returns the IR format of the selected stage for a given function `func_name`
// using library runtime `runtime` on a platform `platform_name` with given
// `inputs_arg_shape_and_dtype` and `input_handles`.
absl::StatusOr<std::string> GetCompilerIr(
    IrExportStage stage, ProcessFunctionLibraryRuntime* pflr,
    absl::string_view func_name, absl::string_view platform_name,
    EagerContext* context,
    absl::Span<const ArgShapeAndDType> input_arg_shape_and_dtype,
    absl::Span<const TensorHandle* const> input_handles,
    CompilerArgSource compiler_arg_source);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_GET_COMPILER_IR_H_
