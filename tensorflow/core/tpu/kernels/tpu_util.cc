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
#include "tensorflow/core/tpu/kernels/tpu_util.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/platform/random.h"

namespace tensorflow {
namespace tpu {

std::string SessionNameFromMetadata(const SessionMetadata* session_metadata) {
  return session_metadata ? session_metadata->name() : "";
}

std::string ProtoKeyForComputation(const std::string& key, int core) {
  return absl::StrCat(key, ":", core);
}

xla::StatusOr<TpuCompilationCacheKey> ParseCompilationCacheKey(
    const std::string& key) {
  const std::vector<std::string> splits = absl::StrSplit(key, '|');
  if (splits.size() == 1) {
    // No guaranteed_const.
    return TpuCompilationCacheKey(key);
  } else if (splits.size() != 3) {
    return errors::InvalidArgument("Invalid TPU compilation cache key:", key);
  }

  TpuCompilationCacheKey parsed_key(splits.at(0));
  parsed_key.has_guaranteed_const = true;
  parsed_key.session_handle = splits.at(1);
  const string fingerprint = splits.at(2);
  parsed_key.guaranteed_const_fingerprint = [fingerprint] {
    return fingerprint;
  };
  return parsed_key;
}

xla::CompileOnlyClient::AotXlaComputationInstance
BuildAotXlaComputationInstance(
    const XlaCompiler::CompilationResult& compilation_result) {
  xla::CompileOnlyClient::AotXlaComputationInstance instance;
  instance.computation = compilation_result.computation.get();
  for (const xla::Shape& shape : compilation_result.xla_input_shapes) {
    instance.argument_layouts.push_back(&shape);
  }
  instance.result_layout = &compilation_result.xla_output_shape;
  return instance;
}

Status ShapeTensorToTensorShape(const Tensor& tensor, TensorShape* shape) {
  if (tensor.dtype() != DT_INT64 ||
      !TensorShapeUtils::IsVector(tensor.shape())) {
    return errors::InvalidArgument("Shape tensor must be an int64 vector.");
  }
  const int64 rank = tensor.NumElements();
  auto tensor_dims = tensor.flat<int64>();
  std::vector<int64> dims(rank);
  for (int64 i = 0; i < rank; ++i) {
    dims[i] = tensor_dims(i);
  }
  return TensorShapeUtils::MakeShape(dims, shape);
}

Status DynamicShapesToTensorShapes(const OpInputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes) {
  shapes->resize(dynamic_shapes.size());
  for (int i = 0; i < dynamic_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(
        ShapeTensorToTensorShape(dynamic_shapes[i], &(*shapes)[i]));
  }
  return Status::OK();
}

Status DynamicShapesToTensorShapes(const InputList& dynamic_shapes,
                                   std::vector<TensorShape>* shapes) {
  shapes->resize(dynamic_shapes.end() - dynamic_shapes.begin());
  size_t i = 0;
  for (auto& dynamic_shape : dynamic_shapes) {
    TF_RETURN_IF_ERROR(
        ShapeTensorToTensorShape(dynamic_shape.tensor(), &(*shapes)[i]));
    ++i;
  }
  return Status::OK();
}
}  // namespace tpu
}  // namespace tensorflow
