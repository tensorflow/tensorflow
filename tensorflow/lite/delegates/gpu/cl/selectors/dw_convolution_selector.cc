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
#include "tensorflow/lite/delegates/gpu/cl/cl_device.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/depth_wise_conv_3x3.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

Status SelectDWConvolutionAdreno(const DepthwiseConvolution2DAttributes& attr,
                                 const CreationContext& creation_context,
                                 const OperationDef& op_def,
                                 std::unique_ptr<GPUOperation>* ptr) {
  if (!op_def.IsBatchSupported() && IsDepthWiseConv3x3Supported(attr)) {
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

Status SelectDWConvolutionPowerVR(const DepthwiseConvolution2DAttributes& attr,
                                 const CreationContext& creation_context,
                                 const OperationDef& op_def,
                                 std::unique_ptr<GPUOperation>* ptr) {
  if (!op_def.IsBatchSupported() && IsDepthWiseConv3x3Supported(attr)) {
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

Status SelectDWConvolutionMali(const DepthwiseConvolution2DAttributes& attr,
                                 const CreationContext& creation_context,
                                 const OperationDef& op_def,
                                 std::unique_ptr<GPUOperation>* ptr) {
  DepthWiseConvolution dw_conv;
  RETURN_IF_ERROR(
      CreateDepthWiseConvolution(creation_context, op_def, attr, &dw_conv));
  *ptr = absl::make_unique<DepthWiseConvolution>(std::move(dw_conv));
  return OkStatus();
}
}  // namespace

Status SelectDWConvolution(const DepthwiseConvolution2DAttributes& attr,
                           const CreationContext& creation_context,
                           const OperationDef& op_def,
                           std::unique_ptr<GPUOperation>* ptr) {
  switch (creation_context.device->vendor()) {
    case Vendor::QUALCOMM:
      return SelectDWConvolutionAdreno(attr, creation_context, op_def, ptr);
    case Vendor::POWERVR:
      return SelectDWConvolutionPowerVR(attr, creation_context, op_def, ptr);
    case Vendor::MALI:
      return SelectDWConvolutionMali(attr, creation_context, op_def, ptr);
    default:
      return SelectDWConvolutionAdreno(attr, creation_context, op_def, ptr);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
