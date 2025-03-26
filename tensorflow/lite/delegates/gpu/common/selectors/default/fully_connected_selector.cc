/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/conv_generic.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/fully_connected.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def, int batch_size) {
  if (op_def.IsBatchSupported()) {
    BHWC dst_shape = BHWC(batch_size, 1, 1, attr.weights.shape.o);
    ConvGeneric conv = CreateConvGeneric(gpu_info, op_def, attr, &dst_shape);
    return std::make_unique<ConvGeneric>(std::move(conv));
  } else {
    FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
    return std::make_unique<FullyConnected>(std::move(fc));
  }
}

std::unique_ptr<GPUOperation> SelectFullyConnected(
    const FullyConnectedInt8Attributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  FullyConnected fc = CreateFullyConnected(gpu_info, op_def, attr);
  return std::make_unique<FullyConnected>(std::move(fc));
}

}  // namespace gpu
}  // namespace tflite
