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
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/fully_connected_texture.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

Status SelectFullyConnected(const FullyConnectedAttributes& attr,
                            const CreationContext& creation_context,
                            const OperationDef& op_def, int batch_size,
                            std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.batch_support) {
    ConvTexture conv;
    RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  } else {
    FullyConnectedTexture fc;
    RETURN_IF_ERROR(
        CreateFullyConnectedTexture(creation_context, op_def, attr, &fc));
    *ptr = absl::make_unique<FullyConnectedTexture>(std::move(fc));
  }
  return OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
