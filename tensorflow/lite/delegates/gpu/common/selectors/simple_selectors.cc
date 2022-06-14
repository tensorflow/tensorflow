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

#include "tensorflow/lite/delegates/gpu/common/selectors/simple_selectors.h"

#include <memory>
#include <set>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/add.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/cast.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_xy.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/concat_z.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/gather.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/lstm.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/padding.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/pooling.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/prelu.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/quantize_and_dequantize.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reduce.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resampler.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reshape.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/reshapex4.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/resize.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/softmax1x1.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/space_to_depth.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/split.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/strided_slice.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/tile.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/transpose.h"
#include "tensorflow/lite/delegates/gpu/common/tasks/winograd.h"

namespace tflite {
namespace gpu {

std::unique_ptr<GPUOperation> SelectLSTM(const OperationDef& op_def,
                                         const GpuInfo& gpu_info) {
  return std::make_unique<GPUOperation>(CreateLSTM(op_def, gpu_info));
}

std::unique_ptr<GPUOperation> SelectReLU(const ReLUAttributes& attr,
                                         const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreateReLU(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPReLU(const PReLUAttributes& attr,
                                          const GpuInfo& gpu_info,
                                          const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreatePReLU(gpu_info, op_def, attr));
}

std::unique_ptr<GPUOperation> SelectPooling(const Pooling2DAttributes& attr,
                                            const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreatePooling(op_def, attr));
}

std::unique_ptr<GPUOperation> SelectMaxUnpooling(
    const MaxUnpooling2DAttributes& attr, const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(CreateMaxUnpooling(op_def, attr));
}

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateAdd(op_def, channels, dst_channels);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectGather(const GatherAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
  if (attr.axis != Axis::WIDTH) {
    return absl::UnimplementedError(
        "No gather for this axis. Only Width axis supported.");
  }
  GPUOperation operation = CreateGather(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
  return absl::OkStatus();
}

std::unique_ptr<GPUOperation> SelectResampler(const OperationDef& op_def,
                                              const GpuInfo& gpu_info) {
  GPUOperation operation = CreateResampler(gpu_info, op_def);
  return std::make_unique<GPUOperation>(std::move(operation));
}

absl::Status SelectResize(const Resize2DAttributes& attr,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
  Resize operation = CreateResize(op_def, attr);
  *ptr = std::make_unique<Resize>(std::move(operation));
  return absl::OkStatus();
}

absl::Status SelectConcat(const ConcatAttributes& attr,
                          const std::vector<int>& channels,
                          const OperationDef& op_def, const GpuInfo& gpu_info,
                          std::unique_ptr<GPUOperation>* ptr) {
  switch (attr.axis) {
    case Axis::CHANNELS: {
      GPUOperation operation = CreateConcatZ(op_def, channels, gpu_info);
      *ptr = std::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    case Axis::BATCH:
    case Axis::DEPTH:
    case Axis::HEIGHT:
    case Axis::WIDTH: {
      GPUOperation operation = CreateConcatXY(op_def, attr);
      *ptr = std::make_unique<GPUOperation>(std::move(operation));
      return absl::OkStatus();
    }
    default:
      return absl::UnimplementedError("No concat for this axis.");
  }
}

std::unique_ptr<GPUOperation> SelectDWConvolutionDynamicWeights(
    const DepthwiseConvolution2DAttributes& attr, const GpuInfo& gpu_info,
    const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateDepthwiseConvolution2DDynamicWeights(gpu_info, op_def, attr));
}

void SelectReshape(int src_channels, int dst_channels,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  if (src_channels % 4 == 0 && dst_channels % 4 == 0) {
    GPUOperation operation = CreateReshapex4(op_def);
    *ptr = std::make_unique<GPUOperation>(std::move(operation));
  } else {
    GPUOperation operation = CreateReshape(op_def);
    *ptr = std::make_unique<GPUOperation>(std::move(operation));
  }
}

void SelectSpaceToDepth(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateSpaceToDepth(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectDepthToSpace(const SpaceToDepthAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateDepthToSpace(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectSplit(const SplitAttributes& attr, const GpuInfo& gpu_info,
                 const std::vector<int>& channels, const OperationDef& op_def,
                 std::unique_ptr<GPUOperation>* ptr) {
  Split operation = CreateSplit(gpu_info, op_def, attr, channels);
  *ptr = std::make_unique<Split>(std::move(operation));
}

void SelectPadding(const PadAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreatePadding(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

void SelectStridedSlice(const SliceAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  StridedSlice operation = CreateStridedSlice(op_def, attr);
  *ptr = std::make_unique<StridedSlice>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectReduce(const std::set<Axis>& axis_to_reduce,
                                           const BHWC& src_shape,
                                           OperationType op_type,
                                           const OperationDef& op_def,
                                           const GpuInfo& gpu_info) {
  return std::make_unique<Reduce>(
      CreateReduce(axis_to_reduce, src_shape, op_type, op_def, gpu_info));
}

void SelectSoftmax(const BHWC& shape, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  if (shape.w == 1 && shape.h == 1) {
    Softmax1x1 operation = CreateSoftmax1x1(op_def);
    *ptr = std::make_unique<Softmax1x1>(std::move(operation));
  } else {
    GPUOperation operation = CreateSoftmax(op_def);
    *ptr = std::make_unique<GPUOperation>(std::move(operation));
  }
}

std::unique_ptr<GPUOperation> SelectTile(const OperationDef& op_def,
                                         const BHWC& src_shape) {
  return std::make_unique<GPUOperation>(CreateTile(op_def, src_shape.c));
}

void SelectTranspose(const TransposeAttributes& attr,
                     const OperationDef& op_def,
                     std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateTranspose(op_def, attr);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

std::unique_ptr<GPUOperation> SelectWinograd4x4To36(
    const GpuInfo& gpu_info, const Padding2D& padding,
    const OperationDef& op_def) {
  if (gpu_info.IsApple() || gpu_info.IsAMD()) {
    Winograd4x4To36 operation =
        CreateWinograd4x4To36(op_def, padding, gpu_info);
    return std::make_unique<Winograd4x4To36>(std::move(operation));
  }
  return std::make_unique<Winograd4x4To36TileX6>(
      CreateWinograd4x4To36TileX6(gpu_info, op_def, padding));
}

std::unique_ptr<GPUOperation> SelectWinograd36To4x4(
    const GpuInfo& gpu_info, const OperationDef& op_def,
    const tflite::gpu::Tensor<Linear, DataType::FLOAT32>& biases) {
  if (gpu_info.IsApple() || gpu_info.IsAMD()) {
    Winograd36To4x4 operation = CreateWinograd36To4x4(op_def, biases);
    return std::make_unique<Winograd36To4x4>(std::move(operation));
  }
  return std::make_unique<Winograd36To4x4Tile4x1>(
      CreateWinograd36To4x4Tile4x1(gpu_info, op_def, biases));
}

std::unique_ptr<GPUOperation> SelectQuantizeAndDequantize(
    const QuantizeAndDequantizeAttributes& attr, const OperationDef& op_def) {
  return std::make_unique<GPUOperation>(
      CreateQuantizeAndDequantize(op_def, attr));
}

void SelectCast(const OperationDef& op_def, const GpuInfo& gpu_info,
                std::unique_ptr<GPUOperation>* ptr) {
  GPUOperation operation = CreateCast(op_def, gpu_info);
  *ptr = std::make_unique<GPUOperation>(std::move(operation));
}

}  // namespace gpu
}  // namespace tflite
