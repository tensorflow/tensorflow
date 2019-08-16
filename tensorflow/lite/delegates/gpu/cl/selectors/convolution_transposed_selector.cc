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

#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_transposed_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_3x3_thin.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/convolution_transposed_thin.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status SelectConvolutionTransposedTextureArray(
    const ConvolutionTransposedAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvolutionTransposedThinSupported(*creation_context.device, attr)) {
    ConvolutionTransposedThin conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposedThin(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposedThin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3ThinSupported(*creation_context.device,
                                                     attr)) {
    ConvolutionTransposed3x3Thin conv;
    RETURN_IF_ERROR(CreateConvolutionTransposed3x3Thin(creation_context, op_def,
                                                       attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
  } else {
    ConvolutionTransposed conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposed(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed>(std::move(conv));
  }
  return OkStatus();
}

Status SelectConvolutionTransposedTexture2D(
    const ConvolutionTransposedAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvolutionTransposedThinSupported(*creation_context.device, attr)) {
    ConvolutionTransposedThin conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposedThin(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposedThin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3ThinSupported(*creation_context.device,
                                                     attr)) {
    ConvolutionTransposed3x3Thin conv;
    RETURN_IF_ERROR(CreateConvolutionTransposed3x3Thin(creation_context, op_def,
                                                       attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
  } else {
    ConvolutionTransposed conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposed(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed>(std::move(conv));
  }
  return OkStatus();
}

Status SelectConvolutionTransposedBuffer(
    const ConvolutionTransposedAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvolutionTransposedThinSupported(*creation_context.device, attr)) {
    ConvolutionTransposedThin conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposedThin(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposedThin>(std::move(conv));
  } else if (IsConvolutionTransposed3x3ThinSupported(*creation_context.device,
                                                     attr)) {
    ConvolutionTransposed3x3Thin conv;
    RETURN_IF_ERROR(CreateConvolutionTransposed3x3Thin(creation_context, op_def,
                                                       attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed3x3Thin>(std::move(conv));
  } else {
    ConvolutionTransposed conv;
    RETURN_IF_ERROR(
        CreateConvolutionTransposed(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvolutionTransposed>(std::move(conv));
  }
  return OkStatus();
}
}  // namespace

Status SelectConvolutionTransposed(const ConvolutionTransposedAttributes& attr,
                                   const CreationContext& creation_context,
                                   const OperationDef& op_def,
                                   std::unique_ptr<GPUOperation>* ptr) {
  switch (op_def.GetPrimaryStorageType()) {
    case TensorStorageType::TEXTURE_ARRAY:
      return SelectConvolutionTransposedTextureArray(attr, creation_context,
                                                     op_def, ptr);
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return SelectConvolutionTransposedTexture2D(attr, creation_context,
                                                  op_def, ptr);
    case TensorStorageType::BUFFER:
      return SelectConvolutionTransposedBuffer(attr, creation_context, op_def,
                                               ptr);
    default:
      return InternalError("Unknown storage type.");
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
