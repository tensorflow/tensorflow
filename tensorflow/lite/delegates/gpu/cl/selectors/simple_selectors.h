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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_SELECTORS_SIMPLE_SELECTORS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_SELECTORS_SIMPLE_SELECTORS_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/cl/kernels/gpu_operation.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"

namespace tflite {
namespace gpu {
namespace cl {

void SelectLSTM(const OperationDef& op_def, std::unique_ptr<GPUOperation>* ptr);

void SelectReLU(const CreationContext& creation_context,
                const ReLUAttributes& attr, const OperationDef& op_def,
                std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectPReLU(const PReLUAttributes& attr,
                         const CreationContext& creation_context,
                         const OperationDef& op_def,
                         std::unique_ptr<GPUOperation>* ptr);

void SelectPooling(const Pooling2DAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectMaxUnpooling(const MaxUnpooling2DAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr);

void SelectReshape(int src_channels, int dst_channels,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectPadding(const PadAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectStridedSlice(const SliceAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectMean(const MeanAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectSoftmax(const BHWC& shape, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectTranspose(const TransposeAttributes& attr,
                     const OperationDef& op_def,
                     std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectWinograd4x4To36(const CreationContext& creation_context,
                                   const Padding2D& padding,
                                   const OperationDef& op_def,
                                   std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectWinograd36To4x4(
    const CreationContext& creation_context, const OperationDef& op_def,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases,
    std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr,
    const CreationContext& creation_context, const OperationDef& op_def,
    std::unique_ptr<GPUOperation>* ptr);

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_SELECTORS_SIMPLE_SELECTORS_H_
