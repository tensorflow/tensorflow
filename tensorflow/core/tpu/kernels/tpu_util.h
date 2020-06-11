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
#ifndef TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_H_
#define TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_H_

#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/xla/client/compile_only_client.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_key.h"

namespace tensorflow {
namespace tpu {

// Utility to get session_name from `SessionMetadata`. `SessionMetadata` may
// be null.
std::string SessionNameFromMetadata(const SessionMetadata* session_metadata);

// Generates cache proto key for a given computation on a TPU core.
std::string ProtoKeyForComputation(const std::string& key, int core);

// Returns a TpuCompilationCacheKey parsed from given key or an error.
xla::StatusOr<TpuCompilationCacheKey> ParseCompilationCacheKey(
    const std::string& key);

xla::CompileOnlyClient::AotXlaComputationInstance
BuildAotXlaComputationInstance(
    const XlaCompiler::CompilationResult& compilation_result);

// Returns true if TPU compilation is enabled.
bool IsTpuCompilationEnabled();

// Converts an int64 host memory `tensor` to a `shape`.
Status ShapeTensorToTensorShape(const Tensor& tensor, TensorShape* shape);

Status DynamicShapesToTensorShapes(const OpInputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes);
Status DynamicShapesToTensorShapes(const InputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes);

// Given a tensor of `shape` and `type`, as what shape should it be stored on
// the TPU device? This function tranposes or flattens the excessively-padded
// tensors to rank 1, but leaves other tensor shapes alone.
xla::StatusOr<xla::Shape> TpuShapeRepresentation(const TensorShape& shape,
                                                 DataType type,
                                                 bool use_fast_memory);

// A callback called on exit.
void LogAndExit(int code);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_KERNELS_TPU_UTIL_H_
