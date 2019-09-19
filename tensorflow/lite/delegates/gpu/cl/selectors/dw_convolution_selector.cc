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

#include "tensorflow/lite/delegates/gpu/cl/selectors/dw_convolution_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv_3x3.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status SelectDWConvolutionTextureArray(
    const DepthwiseConvolution2DAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (IsDepthWiseConv3x3Supported(attr)) {
    DepthWiseConv3x3 dw_conv;
    RETURN_IF_ERROR(CreateDepthWiseConv3x3(creation_context, op_def,
                                                  attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConv3x3>(std::move(dw_conv));
  } else {
    DepthWiseConvolution dw_conv;
    RETURN_IF_ERROR(
        CreateDepthWiseConvolution(creation_context, op_def, attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConvolution>(std::move(dw_conv));
  }
  return OkStatus();
}

Status SelectDWConvolutionTexture2D(
    const DepthwiseConvolution2DAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (IsDepthWiseConv3x3Supported(attr)) {
    DepthWiseConv3x3 dw_conv;
    RETURN_IF_ERROR(CreateDepthWiseConv3x3(creation_context, op_def,
                                                  attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConv3x3>(std::move(dw_conv));
  } else {
    DepthWiseConvolution dw_conv;
    RETURN_IF_ERROR(
        CreateDepthWiseConvolution(creation_context, op_def, attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConvolution>(std::move(dw_conv));
  }
  return OkStatus();
}

Status SelectDWConvolutionBuffer(const DepthwiseConvolution2DAttributes& attr,
                                 const CreationContext& creation_context,
                                 const OperationDef& op_def,
                                 std::unique_ptr<GPUOperation>* ptr) {
  if (!creation_context.device->IsMali() && IsDepthWiseConv3x3Supported(attr)) {
    DepthWiseConv3x3 dw_conv;
    RETURN_IF_ERROR(
        CreateDepthWiseConv3x3(creation_context, op_def, attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConv3x3>(std::move(dw_conv));
  } else {
    DepthWiseConvolution dw_conv;
    RETURN_IF_ERROR(
        CreateDepthWiseConvolution(creation_context, op_def, attr, &dw_conv));
    *ptr = absl::make_unique<DepthWiseConvolution>(std::move(dw_conv));
  }
  return OkStatus();
}
}  // namespace

Status SelectDWConvolution(const DepthwiseConvolution2DAttributes& attr,
                           const CreationContext& creation_context,
                           const OperationDef& op_def,
                           std::unique_ptr<GPUOperation>* ptr) {
  switch (op_def.GetPrimaryStorageType()) {
    case TensorStorageType::TEXTURE_ARRAY:
      return SelectDWConvolutionTextureArray(attr, creation_context, op_def,
                                             ptr);
    case TensorStorageType::TEXTURE_2D:
    case TensorStorageType::SINGLE_TEXTURE_2D:
      return SelectDWConvolutionTexture2D(attr, creation_context, op_def, ptr);
    case TensorStorageType::BUFFER:
      return SelectDWConvolutionBuffer(attr, creation_context, op_def, ptr);
    default:
      return InternalError("Unknown storage type.");
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
