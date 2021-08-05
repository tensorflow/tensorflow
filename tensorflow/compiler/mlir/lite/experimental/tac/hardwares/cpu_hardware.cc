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

#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
// CPU
constexpr float kCPUArithmeticUnitCost = 1.0;

// This basically assumes pure load/store. This is just fake data.
constexpr float kCPUCopyUnitCost = 0.5;

// Default values.
constexpr float kCPUDefaultFixedValuedCost = 10000.0;

// Quantized inference cost efficiency.
// For CPU, quantized inference is ~3x faster than the float alternative, this
// is just an estimation.
constexpr float kQuantizedInferenceEfficiency = 0.3;

inline float InferenceTypeEfficiency(InferenceType inference_type) {
  if (inference_type == QUANTIZED_INT8 || inference_type == QUANTIZED_UINT8) {
    return kQuantizedInferenceEfficiency;
  }
  return 1.0;
}

// CPU hardware class which handles CPU capabilities in TFLite.
// This is used by TAC to get op supported/ op cost estimates on CPU.
class CpuHardware : public TargetHardware {
 public:
  // String Identifier for CPU hardware.
  static constexpr char kId[] = "CPU";

  double GetHardwareSwitchingCost(const TargetHardware* from,
                                  size_t buffer_size) const override {
    auto from_type = from->GetTypeId();
    auto to_type = GetTypeId();
    if (from_type == to_type) return 0.0f;

    // TODO(renjieliu): Implement a better version for different hardware cases.
    return buffer_size * kCrossHardwareTransferPerByteCost / 8.0 +
           kCrossHardwareTransferFixedCost;
  }

  mlir::OwningRewritePatternList GetTransformations(
      MLIRContext* context) const override {
    return {context};
  }

  mlir::TypeID GetTypeId() const override {
    return mlir::TypeID::get<CpuHardware>();
  }

  bool IsOpSupported(mlir::Operation* op) const override {
    // All ops in TFL dialect are supported on CPU.
    if (op->getDialect() == nullptr) return false;
    if (op->getDialect()->getNamespace() != "tfl") return false;
    return true;
  }
};

constexpr char CpuHardware::kId[];  // Define kId.

std::unique_ptr<TargetHardware> CreateCpuHardware() {
  return std::make_unique<CpuHardware>();
}

TargetHardwareRegistration<CpuHardware> cpu_hardware("Target device for CPU",
                                                     CreateCpuHardware);

#define TAC_REGISTER_CPU_OP(Op, Create)                                    \
  TargetHardwareOpRegistration<CpuHardware, Op> Op##_CpuHardware_hardware( \
      Create);

// Operation costs on CPU

// Currently used for these ops:
// tfl.conv_2d / tfl.depthwise_conv_2d / tfl.fully_connected
class CpuConvOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    float cost = 0.0;
    int64_t arithmetic_count;
    if (ArithmeticCountUtilHelper::GetArithmeticCountForConvAndFullyconnectedOp(
            op, &arithmetic_count)) {
      cost = arithmetic_count * kCPUArithmeticUnitCost;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override { return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConvOp() {
  return std::make_unique<CpuConvOp>();
}

// Currently used for these ops:
// tfl.Add / tfl.mul
class CpuArithmeticOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    float cost = 0.0;
    int64_t count;
    if (ArithmeticCountUtilHelper::GetFirstOutputCount(op, &count)) {
      cost = kCPUArithmeticUnitCost * count;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override { return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateArithmeticOp() {
  return std::make_unique<CpuArithmeticOp>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape / tfl.pack
class CpuConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    float cost = 0.0;
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count)) {
      cost = kCPUCopyUnitCost * count;
    } else {
      cost = kCPUDefaultFixedValuedCost;
    }
    return cost * InferenceTypeEfficiency(GetInferenceType(op));
  }

  bool IsOpSupported(mlir::Operation* op) const override { return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<CpuConcatOp>();
}

TAC_REGISTER_CPU_OP(Conv2DOp, CreateConvOp);
TAC_REGISTER_CPU_OP(DepthwiseConv2DOp, CreateConvOp);
TAC_REGISTER_CPU_OP(FullyConnectedOp, CreateConvOp);
TAC_REGISTER_CPU_OP(AddOp, CreateArithmeticOp);
TAC_REGISTER_CPU_OP(MulOp, CreateArithmeticOp);
TAC_REGISTER_CPU_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_CPU_OP(ReshapeOp, CreateConcatOp);
TAC_REGISTER_CPU_OP(PackOp, CreateConcatOp);

#undef TAC_REGISTER_CPU_OP
}  // namespace
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
