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

#include "tensorflow/lite/delegates/gpu/cl/selectors/simple_selectors.h"

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/apply_mask.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_xy.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/concat_z.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/lstm.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/max_unpooling.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/multiply_add.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/padding.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/pooling.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/prelu.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/relu.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/reshape.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/reshapex4.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/resize.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/softmax.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/softmax1x1.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/strided_slice.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/transpose.h"

namespace tflite {
namespace gpu {
namespace cl {

void SelectApplyMask(const OperationDef& op_def, const BHWC& src_shape,
                     const BHWC& mask_shape,
                     std::unique_ptr<GPUOperation>* ptr) {
  ApplyMask operation = CreateApplyMask(op_def, src_shape, mask_shape);
  *ptr = absl::make_unique<ApplyMask>(std::move(operation));
}

void SelectLSTM(const OperationDef& op_def,
                std::unique_ptr<GPUOperation>* ptr) {
  LSTM operation = CreateLSTM(op_def);
  *ptr = absl::make_unique<LSTM>(std::move(operation));
}

void SelectReLU(const CreationContext& creation_context,
                const ReLUAttributes& attr, const OperationDef& op_def,
                std::unique_ptr<GPUOperation>* ptr) {
  ReLU relu = CreateReLU(creation_context, op_def, attr);
  *ptr = absl::make_unique<ReLU>(std::move(relu));
}

Status SelectPReLU(const PReLUAttributes& attr,
                   const CreationContext& creation_context,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  PReLU operation;
  RETURN_IF_ERROR(CreatePReLU(creation_context, op_def, attr, &operation));
  *ptr = absl::make_unique<PReLU>(std::move(operation));
  return OkStatus();
}

void SelectPooling(const Pooling2DAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  Pooling pooling = CreatePooling(op_def, attr);
  *ptr = absl::make_unique<Pooling>(std::move(pooling));
}

void SelectMaxUnpooling(const MaxUnpooling2DAttributes& attr,
                        const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  MaxUnpooling operation = CreateMaxUnpooling(op_def, attr);
  *ptr = absl::make_unique<MaxUnpooling>(std::move(operation));
}

void SelectAdd(const OperationDef& op_def, const std::vector<int>& channels,
               int dst_channels, std::unique_ptr<GPUOperation>* ptr) {
  Add operation = CreateAdd(op_def, channels, dst_channels);
  *ptr = absl::make_unique<Add>(std::move(operation));
}

Status SelectResize(const Resize2DAttributes& attr, const OperationDef& op_def,
                    std::unique_ptr<GPUOperation>* ptr) {
  Resize operation = CreateResize(op_def, attr);
  *ptr = absl::make_unique<Resize>(std::move(operation));
  return OkStatus();
}

Status SelectConcat(const ConcatAttributes& attr,
                    const std::vector<int>& channels,
                    const OperationDef& op_def,
                    std::unique_ptr<GPUOperation>* ptr) {
  switch (attr.axis) {
    case Axis::CHANNELS: {
      ConcatZ operation = CreateConcatZ(op_def, channels);
      *ptr = absl::make_unique<ConcatZ>(std::move(operation));
      return OkStatus();
    }
    case Axis::WIDTH:
    case Axis::HEIGHT: {
      ConcatXY operation = CreateConcatXY(op_def, attr, channels.size());
      *ptr = absl::make_unique<ConcatXY>(std::move(operation));
      return OkStatus();
    }
    default:
      return UnimplementedError("No concat for this axis.");
  }
}

void SelectReshape(int src_channels, int dst_channels,
                   const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  if (src_channels % 4 == 0 && dst_channels % 4 == 0) {
    Reshapex4 operation = CreateReshapex4(op_def);
    *ptr = absl::make_unique<Reshapex4>(std::move(operation));
  } else {
    Reshape operation = CreateReshape(op_def);
    *ptr = absl::make_unique<Reshape>(std::move(operation));
  }
}

void SelectPadding(const PadAttributes& attr, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  Padding operation = CreatePadding(op_def, attr);
  *ptr = absl::make_unique<Padding>(std::move(operation));
}

void SelectStridedSlice(const SliceAttributes& attr, const OperationDef& op_def,
                        std::unique_ptr<GPUOperation>* ptr) {
  StridedSlice operation = CreateStridedSlice(op_def, attr);
  *ptr = absl::make_unique<StridedSlice>(std::move(operation));
}

Status SelectMultiplyScalar(const MultiplyScalarAttributes& attr,
                            const CreationContext& creation_context,
                            const OperationDef& op_def,
                            std::unique_ptr<GPUOperation>* ptr) {
  MultiplyAdd operation;
  RETURN_IF_ERROR(
      CreateMultiplyAdd(creation_context, op_def, attr, &operation));
  *ptr = absl::make_unique<MultiplyAdd>(std::move(operation));
  return OkStatus();
}

Status SelectBroadcastAdd(const AddAttributes& attr,
                          const CreationContext& creation_context,
                          const OperationDef& op_def,
                          std::unique_ptr<GPUOperation>* ptr) {
  MultiplyAdd operation;
  RETURN_IF_ERROR(
      CreateMultiplyAdd(creation_context, op_def, attr, &operation));
  *ptr = absl::make_unique<MultiplyAdd>(std::move(operation));
  return OkStatus();
}

void SelectSoftmax(const BHWC& shape, const OperationDef& op_def,
                   std::unique_ptr<GPUOperation>* ptr) {
  if (shape.w == 1 && shape.h == 1) {
    Softmax1x1 operation = CreateSoftmax1x1(op_def);
    *ptr = absl::make_unique<Softmax1x1>(std::move(operation));
  } else {
    Softmax operation = CreateSoftmax(op_def);
    *ptr = absl::make_unique<Softmax>(std::move(operation));
  }
}

void SelectTranspose(const TransposeAttributes& attr,
                     const OperationDef& op_def,
                     std::unique_ptr<GPUOperation>* ptr) {
  Transpose operation = CreateTranspose(op_def, attr);
  *ptr = absl::make_unique<Transpose>(std::move(operation));
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
