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

#include "tensorflow/lite/delegates/gpu/cl/selectors/convolution_selector.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_buffer_1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_constants.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_powervr.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/conv_texture.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/work_group_picking.h"
#include "tensorflow/lite/delegates/gpu/cl/tensor_type.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"

namespace tflite {
namespace gpu {
namespace cl {
namespace {

absl::Status SelectConvolutionAdreno(const Convolution2DAttributes& attr,
                                     const BHWC& dst_shape,
                                     const CreationContext& creation_context,
                                     const OperationDef& op_def,
                                     ModelHints hints,
                                     std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(*creation_context.device, op_def, attr)) {
    ConvConstants conv;
    RETURN_IF_ERROR(CreateConvConstants(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvTexture conv;
    RETURN_IF_ERROR(CreateConvTexture(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionWinogradAdreno(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const CreationContext& creation_context, const OperationDef& op_def,
    ModelHints hints, std::unique_ptr<GPUOperation>* ptr) {
  ConvTexture conv;
  RETURN_IF_ERROR(
      CreateConvTextureWino4x4To6x6(creation_context, op_def, attr, &conv));
  *ptr = absl::make_unique<ConvTexture>(std::move(conv));
  return absl::OkStatus();
}

absl::Status SelectConvolutionNVidia(const Convolution2DAttributes& attr,
                                     const CreationContext& creation_context,
                                     const OperationDef& op_def,
                                     std::unique_ptr<GPUOperation>* ptr) {
  if (IsConvConstantsSupported(*creation_context.device, op_def, attr)) {
    ConvConstants conv;
    RETURN_IF_ERROR(CreateConvConstants(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvConstants>(std::move(conv));
  } else {
    ConvPowerVR conv;
    RETURN_IF_ERROR(CreateConvPowerVR(creation_context, op_def, attr, &conv));
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionPowerVR(const Convolution2DAttributes& attr,
                                      const CreationContext& creation_context,
                                      const OperationDef& op_def,
                                      std::unique_ptr<GPUOperation>* ptr) {
  ConvPowerVR conv;
  RETURN_IF_ERROR(CreateConvPowerVR(creation_context, op_def, attr, &conv));
  *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  return absl::OkStatus();
}

absl::Status SelectConvolutionMali(const Convolution2DAttributes& attr,
                                   const BHWC& dst_shape,
                                   const CreationContext& creation_context,
                                   const OperationDef& op_def,
                                   std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER &&
      IsConvBuffer1x1Supported(op_def, attr)) {
    ConvBuffer1x1 conv;
    RETURN_IF_ERROR(
        CreateConvBuffer1x1(creation_context, op_def, attr, &conv, &dst_shape));
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv;
    RETURN_IF_ERROR(
        CreateConvPowerVR(creation_context, op_def, attr, &conv, &dst_shape));
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

absl::Status SelectConvolutionWinogradMali(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr) {
  if (op_def.src_tensors[0].storage_type == TensorStorageType::BUFFER) {
    ConvBuffer1x1 conv;
    RETURN_IF_ERROR(CreateConvBuffer1x1Wino4x4To6x6(creation_context, op_def,
                                                    attr, &conv, &dst_shape));
    *ptr = absl::make_unique<ConvBuffer1x1>(std::move(conv));
  } else {
    ConvPowerVR conv;
    RETURN_IF_ERROR(CreateConvPowerVRWino4x4To6x6(creation_context, op_def,
                                                  attr, &conv, &dst_shape));
    *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status SelectConvolution(const Convolution2DAttributes& attr,
                               const BHWC& dst_shape,
                               const CreationContext& creation_context,
                               const OperationDef& op_def, ModelHints hints,
                               std::unique_ptr<GPUOperation>* ptr) {
  switch (creation_context.device->vendor()) {
    case Vendor::QUALCOMM:
      return SelectConvolutionAdreno(attr, dst_shape, creation_context, op_def,
                                     hints, ptr);
    case Vendor::POWERVR:
    case Vendor::AMD:
      return SelectConvolutionPowerVR(attr, creation_context, op_def, ptr);
    case Vendor::NVIDIA:
      return SelectConvolutionNVidia(attr, creation_context, op_def, ptr);
    case Vendor::MALI:
      return SelectConvolutionMali(attr, dst_shape, creation_context, op_def,
                                   ptr);
    default:
      return SelectConvolutionAdreno(attr, dst_shape, creation_context, op_def,
                                     hints, ptr);
  }
}

absl::Status SelectConvolutionForWinograd(
    const Convolution2DAttributes& attr, const BHWC& dst_shape,
    const CreationContext& creation_context, const OperationDef& op_def,
    ModelHints hints, std::unique_ptr<GPUOperation>* ptr) {
  switch (creation_context.device->vendor()) {
    case Vendor::QUALCOMM:
      return SelectConvolutionWinogradAdreno(attr, dst_shape, creation_context,
                                             op_def, hints, ptr);
    case Vendor::POWERVR:
    case Vendor::AMD:
    case Vendor::NVIDIA: {
      ConvPowerVR conv;
      RETURN_IF_ERROR(
          CreateConvPowerVRWino4x4To6x6(creation_context, op_def, attr, &conv));
      *ptr = absl::make_unique<ConvPowerVR>(std::move(conv));
      return absl::OkStatus();
    }
    case Vendor::MALI:
      return SelectConvolutionWinogradMali(attr, dst_shape, creation_context,
                                           op_def, ptr);
    default:
      return SelectConvolutionWinogradAdreno(attr, dst_shape, creation_context,
                                             op_def, hints, ptr);
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
