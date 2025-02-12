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

#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/nnapi_hardware.h"

#include <cstdint>
#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform_patterns.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/arithmetic_count_util.h"

namespace mlir {
namespace TFL {
namespace tac {

// The copy can be non-consectutive copy. This is just fake data.
constexpr float kNNAPICopyUnitCost = 0.2;

// Default values.
constexpr float kNNAPIDefaultFixedValuedCost = 10000.0;

constexpr char NNAPIHardware::kId[];  // Define kId.

mlir::RewritePatternSet NNAPIHardware::GetTransformations(
    MLIRContext* context) const {
  mlir::RewritePatternSet patterns(context);

  patterns.add<SquaredDifference, LowerPackIntoConcatReshape,
               ReduceMeanToAvgPool, InsertRequantForReduceMean>(context);
  return patterns;
}

std::unique_ptr<TargetHardware> CreateNNAPIHardware() {
  return std::make_unique<NNAPIHardware>();
}

TargetHardwareRegistration<NNAPIHardware> nnapi_hardware(
    "Target device for NNAPI", CreateNNAPIHardware);

// Currently used for these ops:
// tfl.squared_difference
class NNAPIBasicSupportedOpNoCost : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override { return 0; }

  bool IsOpSupported(mlir::Operation* op) const override {
    return true;
  }
};

std::unique_ptr<TargetHardwareOperation> CreateBasicOpNoCost() {
  return std::make_unique<NNAPIBasicSupportedOpNoCost>();
}

// Currently used for these ops:
// tfl.concatenation / tfl.reshape / tfl.pack
class NNAPIConcatOp : public TargetHardwareOperation {
  double GetOpCost(mlir::Operation* op) const override {
    int64_t count;
    if (ArithmeticCountUtilHelper::GetInputTensorTotalSize(op, &count))
      return kNNAPICopyUnitCost * count;
    return kNNAPIDefaultFixedValuedCost;
  }

  bool IsOpSupported(mlir::Operation* op) const override { return true; }
};
std::unique_ptr<TargetHardwareOperation> CreateConcatOp() {
  return std::make_unique<NNAPIConcatOp>();
}

#define TAC_REGISTER_NNAPI_OP(Op, Create)                                      \
  TargetHardwareOpRegistration<NNAPIHardware, Op> Op##_NNAPIHardware_hardware( \
      Create);

// Op registeration
TAC_REGISTER_NNAPI_OP(SquaredDifferenceOp, CreateBasicOpNoCost);
TAC_REGISTER_NNAPI_OP(ConcatenationOp, CreateConcatOp);
TAC_REGISTER_NNAPI_OP(ReshapeOp, CreateConcatOp);
TAC_REGISTER_NNAPI_OP(PackOp, CreateConcatOp);

#undef TAC_REGISTER_NNAPI_OP
}  // namespace tac
}  // namespace TFL
}  // namespace mlir
