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
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"

namespace tensorflow {
namespace tpu {

namespace {

std::string CreateShapePrefix(
    const std::vector<tensorflow::TensorShape>& dynamic_shapes) {
  std::string shapes_prefix;
  for (const TensorShape& shape : dynamic_shapes) {
    for (int64_t size : shape.dim_sizes()) {
      absl::StrAppend(&shapes_prefix, size, ",");
    }
    absl::StrAppend(&shapes_prefix, ";");
  }
  return shapes_prefix;
}

}  // namespace

uint64_t CreateFingerprintWithNameAndShapes(
    uint64_t name, const std::vector<tensorflow::TensorShape>& shapes) {
  std::string shape_prefix = CreateShapePrefix(shapes);
  VLOG(2) << "CreateFingerprintWithNameAndShapes, name: " << name
          << ", shape_prefix: " << shape_prefix;
  return TpuCompileInterface::Get()->FingerprintString(
      absl::StrCat(name, "_", shape_prefix));
}

}  // namespace tpu
}  // namespace tensorflow
