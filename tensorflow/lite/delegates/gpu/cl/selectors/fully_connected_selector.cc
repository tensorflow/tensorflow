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

#include "tensorflow/lite/delegates/gpu/cl/selectors/fully_connected_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Status SelectFullyConnectedAdreno(const FullyConnectedAttributes& attr,
                                  const CreationContext& creation_context,
                                  const OperationDef& op_def, int batch_size,
                                  std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.IsBatchSupported()) {
    ConvTexture conv;
    RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  } else {
    FullyConnected fc;
    RETURN_IF_ERROR(
        CreateFullyConnected(creation_context, op_def, attr, &fc));
    *ptr = absl::make_unique<FullyConnected>(std::move(fc));
  }
  return OkStatus();
}

Status SelectFullyConnectedPowerVR(const FullyConnectedAttributes& attr,
                                   const CreationContext& creation_context,
                                   const OperationDef& op_def, int batch_size,
                                   std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.IsBatchSupported()) {
    ConvPowerVR conv;
    RETURN_IF_ERROR(CreateConvPowerVR(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  } else {
    FullyConnected fc;
    RETURN_IF_ERROR(
        CreateFullyConnected(creation_context, op_def, attr, &fc));
    *ptr = absl::make_unique<FullyConnected>(std::move(fc));
  }
  return OkStatus();
}

Status SelectFullyConnectedMali(const FullyConnectedAttributes& attr,
                                const CreationContext& creation_context,
                                const OperationDef& op_def, int batch_size,
                                std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.IsBatchSupported()) {
    if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
      ConvBuffer1x1 conv;
      RETURN_IF_ERROR(
          CreateConvBuffer1x1(creation_context, op_def, attr, &conv));
      *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
    } else {
      ConvTexture conv;
      RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
      *ptr = absl::make_unique<ConvTexture>(std::move(conv));
    }
  } else {
    FullyConnected fc;
    RETURN_IF_ERROR(
        CreateFullyConnected(creation_context, op_def, attr, &fc));
    *ptr = absl::make_unique<FullyConnected>(std::move(fc));
  }
  return OkStatus();
}

Status SelectFullyConnected(const FullyConnectedAttributes& attr,
                            const CreationContext& creation_context,
                            const OperationDef& op_def, int batch_size,
                            std::unique_ptr<GPUOperation>* ptr) {
  switch (creation_context.device->vendor()) {
    case Vendor::QUALCOMM:
      return SelectFullyConnectedAdreno(attr, creation_context, op_def,
                                        batch_size, ptr);
    case Vendor::POWERVR:
      return SelectFullyConnectedPowerVR(attr, creation_context, op_def,
                                         batch_size, ptr);
    case Vendor::MALI:
      return SelectFullyConnectedMali(attr, creation_context, op_def,
                                      batch_size, ptr);
    default:
      return SelectFullyConnectedAdreno(attr, creation_context, op_def,
                                        batch_size, ptr);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
