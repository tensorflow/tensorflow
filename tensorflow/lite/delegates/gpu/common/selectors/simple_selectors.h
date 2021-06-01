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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SIMPLE_SELECTORS_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SIMPLE_SELECTORS_H_

#include <memory>

#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/task/gpu_operation.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectLSTM(const OperationDef& op_def,
                                         const GpuInfo& gpu_info);

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes& attr,
                                         const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectPReLU(const PReLUAttributes& attr,
                                          const GpuInfo& gpu_info,
                                          const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes& attr,
                                            const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectMaxUnpooling(
    const MaxUnpooling2DAttributes& attr, const OperationDef& op_def);

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectGather(const GatherAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr);

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectResampler(const OperationDef& op_def);

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def, const GpuInfo& gpu_info,
                          std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectDWConvolutionDynamicWeights(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def);

void SelectReshape(int src_channels, int dst_channels,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectPadding(const PadAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectStridedSlice(const SliceAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectReduce(const std::set<Axis>& axis_to_reduce,
                                           const BHWC& src_shape,
                                           OperationType op_type,
                                           const OperationDef& op_def,
                                           const GpuInfo& gpu_info);

void SelectSoftmax(const BHWC& shape, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr);

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectDepthToSpace(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr);

void SelectSplit(const SplitAttributes& attr, const OperationDef& op_def,
                 std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectTile(const OperationDef& op_def,
                                         const BHWC& src_shape);

void SelectTranspose(const TransposeAttributes& attr,
                     const OperationDef& op_def,
                     std::unique_ptr<GPUOperation>* ptr);

std::unique_ptr<GPUOperation> SelectWinograd4x4To36(const GpuInfo& gpu_info,
                                                    const Padding2D& padding,
                                                    const OperationDef& op_def);

std::unique_ptr<GPUOperation> SelectWinograd36To4x4(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases);

std::unique_ptr<GPUOperation> SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_SELECTORS_SIMPLE_SELECTORS_H_
