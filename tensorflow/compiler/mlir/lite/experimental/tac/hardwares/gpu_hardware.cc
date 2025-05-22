/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/gpu_hardware.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/generated_transform_patterns.inc"
}  // namespace

constexpr char GpuHardware::kId[];  // Define kId.

mlir::RewritePatternSet GpuHardware::GetTransformations(
    MLIRContext* context) const {
  mlir::RewritePatternSet patterns(context);

  patterns.add<LowerPackIntoConcatReshape, UnrollSplit, UnrollSplitV, SubToAdd,
               EnsureBiasForConv2d, PadSlice, FullyConnectedToConv, PadConcat,
               SquaredDifference>(context);
  return patterns;
}

double GpuHardware::GetHardwareSwitchingCost(const TargetHardware* from,
                                             size_t buffer_size) const {
  auto from_type = from->GetTypeId();
  auto to_type = GetTypeId();
  if (from_type == to_type) return 0.0f;

  // TODO(renjieliu): Implement a better version for different hardware cases.
  return buffer_size * kCrossHardwareTransferPerByteCost / 8.0 +
         kCrossHardwareTransferFixedCost;
}

bool GpuHardware::IsOpSupported(mlir::Operation* op) const {
  if (TargetHardware::IsOpSupported(op)) {
    return true;
  }

  // We also support quantized ops.
  return !NotTFLQuantDequantizeOp(op);
}

namespace {
// GPU
constexpr float kGPUArithmeticUnitCost = 0.2;

// The copy can be non-consectutive copy. This is just fake data.
constexpr float kGPUCopyUnitCost = 0.2;

// Default values.
constexpr float kGPUDefaultFixedValuedCost = 10000.0;

std::unique_ptr<TargetHardware> CreateGpuHardware() {
  return std::make_unique<GpuHardware>();
}

TargetHardwareRegistration<GpuHardware> gpu_hardware("Target device for GPU",
                                                     CreateGpuHardware);

#define TAC_REGISTER_GPU_OP(Op, Create)                                    \
  TargetHardwareOpRegistration<GpuHardware, Op> Op##_GpuHardware_hardware( \
      Create);

// Currently used for these ops:
// tfl.Abs / tfl.Average_pool_2d / tfl.Cos / tfl.div / tfl.exp / tfl.hardswish /
// tfl.log / tfl.logistic / tfl.max_pool_2d / tfl.mirror_pad / tfl.maximum /
// tfl.custom / tfl.mean / tfl.minimum / tfl.pad / tfl.pow / tfl.prelu /
// tfl.relu / tfl.relu6 / tfl.rsqrt / tfl.sin / tfl.slice / tfl.softmax /
// tfl.space_to_depth / tfl.sqrt / tfl.square / tfl.squared_difference /
// tfl.strided_slice / tfl.tanh / tfl.transpose / tfl.transpose_conv
class GpuBasicSupportedOpNoCost : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override { return 0; }

  bool IsOpSupported(mlir::Operation* op) const override {
    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateBasicOpNoCost() {
  return std::make_unique<GpuBasicSupportedOpNoCost>();
}

// Currently used for these ops:
// tfl.Add / tfl.mul
class GpuArithmeticOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count))
      return kGPUArithmeticUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateArithmeticOp() {
  return std::make_unique<GpuArithmeticOp>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape
class GpuConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kGPUCopyUnitCost * count;
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<GpuConcatOp>();
}

// Currently used for these ops:
// tfl.conv_2d / tfl.depthwise_conv_2d / tfl.fully_connected
class GpuConvOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      return arithmetic_count * kGPUArithmeticUnitCost;
    }
    return kGPUDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override {
    InferenceType inference_type = GetInferenceType(op);
    if (inference_type != FLOAT) {
      return false;
    }
    return true;
  }
};
std::unique_ptr<TargetHardwareOperation> CreateConvOp() {
  return std::make_unique<GpuConvOp>();
}

// Op registrations
TAC_REGISTER_GPU_OP(AbsOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(AveragePool2DOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(CosOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(DivOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(ExpOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(HardSwishOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(LogOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(LogisticOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MaxPool2DOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MirrorPadOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MaximumOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MinimumOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(MeanOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(CustomOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PadOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PowOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(PReluOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(ReluOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(Relu6Op, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(RsqrtOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SinOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SliceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SoftmaxOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SpaceToDepthOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SqrtOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SquareOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(SquaredDifferenceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(StridedSliceOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TanhOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TransposeOp, CreateBasicOpNoCost);
TAC_REGISTER_GPU_OP(TransposeConvOp, CreateBasicOpNoCost);

TAC_REGISTER_GPU_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_GPU_OP(ReshapeOp, CreateConcatOp);

TAC_REGISTER_GPU_OP(Conv2DOp, CreateConvOp);
TAC_REGISTER_GPU_OP(DepthwiseConv2DOp, CreateConvOp);
TAC_REGISTER_GPU_OP(FullyConnectedOp, CreateConvOp);

TAC_REGISTER_GPU_OP(AddOp, CreateArithmeticOp);
TAC_REGISTER_GPU_OP(MulOp, CreateArithmeticOp);

#undef TAC_REGISTER_GPU_OP
}  // namespace
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
