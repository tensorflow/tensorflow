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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"

#include <algorithm>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/eval_util.h"
#include "tensorflow/core/platform/mutex.h"

namespace mlir {
namespace TF {

// Implements a TF specific policy on when constant folding is allowed.
// Policy:
//
// Disable constant folding if operands size is greater than a certain
// threshold (`kOperandsSizeThreshold`).
//
// Otherwise, allow folding if we do not know the shape of an operand or
// result i.e., one of these values has non-static shape. If we know all the
// shapes, find the total size of the operands and results. Folding of the op is
// allowed if one of the following conditions are met:
// 1. size of results is less than a certain threshold
// (`kResultsSizeThreshold`), or
// 2. size of results is within a factor (`kSizeFactor`) of size of operands, or
// TODO(b/157226221): Look into other heuristics for constant fold policy.
static bool ShouldBeFolded(Operation* inst) {
  bool has_unknown_shape = false;
  auto get_size = [&](TypeRange types) {
    int64_t size = 0;
    for (auto t : types) {
      auto tensor_type = t.cast<TensorType>();
      // Ignore types with undefined bit widths.
      if (!tensor_type.getElementType().isIntOrFloat()) continue;
      if (!tensor_type.hasStaticShape()) {
        has_unknown_shape = true;
        return size;
      }
      size += tensor_type.getNumElements() *
              tensor_type.getElementType().getIntOrFloatBitWidth();
    }
    return size;
  };

  int64_t results_size = get_size(inst->getResultTypes());
  int64_t operands_size = get_size(inst->getOperandTypes());

  constexpr int kSizeFactor = 2;
// TODO(b/233827625): Remove TF_DISABLE_CONSTANT_FOLDING macro.
#ifdef TF_DISABLE_CONSTANT_FOLDING
  constexpr int64_t kResultsSizeThreshold = 0;
#else
  constexpr int64_t kResultsSizeThreshold = (1 << 23);   // 1 MB
#endif
  constexpr int64_t kOperandsSizeThreshold = (1 << 30);  // 1 GB

  return (operands_size <= kOperandsSizeThreshold) &&
         (has_unknown_shape || (results_size <= kResultsSizeThreshold) ||
          (results_size <= kSizeFactor * operands_size));
}

LogicalResult ConstantFoldFallbackHook(
    Operation* inst, ArrayRef<Attribute> operands,
    SmallVectorImpl<OpFoldResult>& results) {  // NOLINT
  // Instructions with side effects should not be constant folded to preserve
  // the original semantics. Ops that have no side effect and zero results but
  // could be folded should have a custom folder instead of relying on the
  // TensorFlow folding hook.
  if (inst->getNumResults() == 0 ||
      inst->hasTrait<OpTrait::TF::NoConstantFold>() ||
      inst->getNumRegions() != 0 || !MemoryEffectOpInterface::hasNoEffect(inst))
    return failure();

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (auto type : inst->getResultTypes()) {
    if (auto tensor_type = type.dyn_cast<TensorType>()) {
      if (tensor_type.getElementType().isa<VariantType>()) {
        return failure();
      }
    }
  }

  // If all the results are empty and has numerical element types, set results
  // to empty elements attribute. This is restricted to the numerical element
  // types as the DenseElementsAttr only supports numerical and string types.
  // TODO(hinsu): Handle ops that have one of the results empty for constant
  // propagation.
  bool has_empty_numerical_results =
      llvm::all_of(inst->getResultTypes(), [](Type ty) {
        ShapedType shaped_ty = ty.cast<ShapedType>();
        Type element_ty = shaped_ty.getElementType();
        return shaped_ty.hasStaticShape() && shaped_ty.getNumElements() == 0 &&
               element_ty.isIntOrFloat();
      });
  if (has_empty_numerical_results &&
      // TODO(jpienaar): Remove this once some unmodeled op behavior is
      // addressed.
      inst->isRegistered()) {
    for (Type ty : inst->getResultTypes()) {
      auto shaped_ty = ty.cast<ShapedType>();
      results.push_back(
          DenseElementsAttr::get(shaped_ty, llvm::ArrayRef<Attribute>()));
    }
    return success();
  }

  // Do not execute function calls.
  if (llvm::isa<TF::WhileOp, TF::CaseOp, TF::IfOp, CallOpInterface>(inst)) {
    return failure();
  }

  // Determine if we should attempt to fold this operation by considering the
  // size/size increase due to folding.
  if (!ShouldBeFolded(inst)) return failure();

  // TODO(jpienaar): Currently this persists the entire program execution. This
  // should instead be per module/set from the Graph being executed in TF (if
  // any) so that the value of variables in the context could be read.
  // Note: Sharing the context is fine as ops are side-effect free.
  auto initialize = []() -> TFE_Context* {
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)>
        opts(TFE_NewContextOptions(), TFE_DeleteContextOptions);
    // Only initialize single CPU.
    tensorflow::ConfigProto config_proto;
    // This is conceptually equal to what we do in python/eager/context.py but
    // with all GPU devices ignored and CPU only set to 1.
    (*config_proto.mutable_device_count())["CPU"] = 1;
    (*config_proto.mutable_device_count())["GPU"] = 0;
    std::unique_ptr<TF_Buffer, decltype(&TF_DeleteBuffer)> config(
        TF_NewBuffer(), TF_DeleteBuffer);
    DCHECK(config->data == nullptr);

    // Copy config_proto into config.
    {
      const size_t proto_size = config_proto.ByteSizeLong();
      void* buf = tensorflow::port::Malloc(proto_size);
      if (buf == nullptr) {
        LOG(ERROR) << "Failed to allocate memory to serialize ConfigProto "
                      "while creating context options for constant folding";
        return nullptr;
      }
      if (!config_proto.SerializeWithCachedSizesToArray(
              static_cast<uint8_t*>(buf))) {
        tensorflow::port::Free(buf);
        LOG(ERROR) << "Unable to serialize ConfigProto while creating context "
                      "options for constant folding";
        return nullptr;
      }
      config->data = buf;
      config->length = proto_size;
      config->data_deallocator = [](void* data, size_t length) {
        tensorflow::port::Free(data);
      };
    }

    TFE_ContextOptionsSetConfig(opts.get(), config->data, config->length,
                                status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      LOG(ERROR) << "Failed to set context options for constant folding: "
                 << status.get();
      return nullptr;
    }

    // Input tensors are placed on the host CPU so use the explicit device
    // policy to fail if no CPU kernels are available for the op.
    TFE_ContextOptionsSetDevicePlacementPolicy(opts.get(),
                                               TFE_DEVICE_PLACEMENT_EXPLICIT);
    auto ctx = TFE_NewContext(opts.get(), status.get());
    if (TF_GetCode(status.get()) != TF_OK) {
      LOG(ERROR) << "Failed to create context for constant folding: "
                 << status.get();
      return nullptr;
    }
    return ctx;
  };
  static TFE_Context* ctx = initialize();
  if (!ctx) return failure();

  // Returns directly if any of the operands is not an elements attributes.
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !attr.isa<ElementsAttr>();
      }))
    return failure();

  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(input.cast<ElementsAttr>());
  }

  // Avoid overlapping folds with the same context.
  // TODO(jpienaar): Avoid using global context & mutex here.
  static auto* mu = new tensorflow::mutex();
  tensorflow::mutex_lock l(*mu);
  SmallVector<Attribute, 8> constants;
  LogicalResult status =
      tensorflow::EvaluateOperation(inst, inputs, ctx, &constants);
  results.assign(constants.begin(), constants.end());
  return status;
}

static bool init_hooks = ([] () {
  TensorFlowDialect::RegisterConstantFoldHook(ConstantFoldFallbackHook);
}(), true);

}  // namespace TF
}  // namespace mlir
