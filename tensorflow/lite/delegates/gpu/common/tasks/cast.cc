/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/tasks/cast.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/substitute.h"
#include "tensorflow/lite/delegates/gpu/common/task/util.h"

namespace tflite {
namespace gpu {
GPUOperation CreateCast(const OperationDef& definition,
                        const GpuInfo& gpu_info) {
  ElementwiseDescriptor op_desc;
  const std::string conversion =
      GetTypeConversion(gpu_info, definition.src_tensors[0].GetDataType(),
                        definition.dst_tensors[0].GetDataType(), 4);
  op_desc.code =
      "out_value = " + absl::Substitute(conversion, "in_value") + ";\n";
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
