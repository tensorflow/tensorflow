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

#include "tensorflow/compiler/mlir/tensorflow/transforms/decompose_resource_ops.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.h"
#include "tensorflow/core/framework/rng_alg.h"

namespace mlir {
namespace TF {

namespace {

// Returns subtype of `resource` if present. Otherwise an unranked tensor type
// of `element_type` is returned.
static Type GetResourceSubtypeOrDefault(Value resource, Type element_type) {
  auto resource_type = resource.getType()
                           .cast<TensorType>()
                           .getElementType()
                           .cast<ResourceType>();
  if (resource_type.getSubtypes().size() == 1)
    return resource_type.getSubtypes().front();

  return UnrankedTensorType::get(element_type);
}

static bool HasResourceSubtype(Value resource) {
  return resource.getType()
             .cast<TensorType>()
             .getElementType()
             .cast<ResourceType>()
             .getSubtypes()
             .size() == 1;
}

static Type GetResourceSubtype(Value resource) {
  return resource.getType()
      .cast<TensorType>()
      .getElementType()
      .cast<ResourceType>()
      .getSubtypes()
      .front();
}

// Decompose tf.RngReadAndSkip.
//
// For Philox, the resource variable holds a tensor<3xi64> with the state:
//   [counter_lo, counter_hi, key]
//
//   RngReadAndSkip increments the 128 bit counter value by 256 * delta and
//   returns the original state value.
//
// For Threefry, the resource variable holds a tensor<2xi64> with the state:
//   [counter, key]
//
//   RngReadAndSkip increments the 64 bit counter value by 256 * delta and
//   returns a tensor<3xi64> value [counter, key, 0].
class DecomposeRngReadAndSkipOp : public RewritePattern {
 public:
  explicit DecomposeRngReadAndSkipOp(MLIRContext *context)
      : RewritePattern(RngReadAndSkipOp::getOperationName(), 1, context,
                       {
                           AddV2Op::getOperationName(),
                           AssignVariableOp::getOperationName(),
                           CastOp::getOperationName(),
                           ConstOp::getOperationName(),
                           LessOp::getOperationName(),
                           MulOp::getOperationName(),
                           PadOp::getOperationName(),
                           PackOp::getOperationName(),
                           ReadVariableOp::getOperationName(),
                           SelectV2Op::getOperationName(),
                           UnpackOp::getOperationName(),
                       }) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto rng_op = cast<RngReadAndSkipOp>(op);

    DenseIntElementsAttr alg_constant;
    if (!matchPattern(rng_op.alg(), m_Constant(&alg_constant))) {
      return rewriter.notifyMatchFailure(
          op, "unable to determine algorithm statically");
    }

    if (alg_constant.getNumElements() != 1) {
      return rewriter.notifyMatchFailure(op, "expected alg to be a scalar");
    }

    uint64_t alg_value = ((*alg_constant.value_begin<APInt>()).getZExtValue());
    tensorflow::ConcreteRngAlgorithm alg;
    if (tensorflow::RNG_ALG_PHILOX == alg_value) {
      alg = tensorflow::ConcreteRngAlgorithm::RNG_ALG_PHILOX;
    } else if (tensorflow::RNG_ALG_THREEFRY == alg_value) {
      alg = tensorflow::ConcreteRngAlgorithm::RNG_ALG_THREEFRY;
    } else if (tensorflow::RNG_ALG_AUTO_SELECT == alg_value) {
      // For AUTO_SELECT, we'll manage the counter as if it's for Philox.
      alg = tensorflow::ConcreteRngAlgorithm::RNG_ALG_PHILOX;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported alg");
    }

    Type state_element_type = rewriter.getI64Type();
    RankedTensorType op_type = RankedTensorType::get(
        {tensorflow::RNG_MAX_COUNTER_SIZE + tensorflow::RNG_KEY_SIZE},
        state_element_type);
    if (op_type != rng_op.getType()) {
      return rewriter.notifyMatchFailure(op, "unexpected op type");
    }

    if (!HasResourceSubtype(rng_op.resource())) {
      return rewriter.notifyMatchFailure(op, "missing resource subtype");
    }

    int counter_size = tensorflow::GetCounterSize(alg);
    int state_size = counter_size + tensorflow::RNG_KEY_SIZE;
    RankedTensorType res_type =
        RankedTensorType::get({state_size}, state_element_type);
    if (res_type != GetResourceSubtype(rng_op.resource())) {
      return rewriter.notifyMatchFailure(op, "unexpected resource subtype");
    }

    Location loc = op->getLoc();

    // Read the state value from the resource.
    Value state =
        rewriter.create<ReadVariableOp>(loc, res_type, rng_op.resource());

    // Extract the key and counter from the state.
    RankedTensorType word_type = RankedTensorType::get({}, state_element_type);
    auto unpacked = rewriter.create<UnpackOp>(
        loc, SmallVector<Type, 4>(state_size, word_type), state, 0);
    Value key = unpacked.getResult(counter_size);

    SmallVector<Value, 4> counter;
    for (int i = 0; i < counter_size; ++i) {
      counter.push_back(unpacked.getResult(i));
    }

    // Set the increment to 256 * delta.
    Type u64 = rewriter.getIntegerType(64, /*isSigned=*/false);
    RankedTensorType u64_scalar = RankedTensorType::get({}, u64);
    Value step_size = rewriter.create<ConstOp>(loc, GetScalarOfType(u64, 256));
    Value increment =
        rewriter.create<MulOp>(loc, u64_scalar, step_size, rng_op.delta());

    // Increment the counter.
    SmallVector<Value, 4> pack_args;
    RankedTensorType word_u64_type = RankedTensorType::get({}, u64);
    Value zero_u64 = rewriter.create<ConstOp>(loc, GetScalarOfType(u64, 0));
    Value one_u64 = rewriter.create<ConstOp>(loc, GetScalarOfType(u64, 1));
    for (int i = 0; i < counter_size; ++i) {
      Value word = counter[i];
      Value word_u64 = rewriter.create<CastOp>(loc, word_u64_type, word);
      Value new_word_u64 = rewriter.create<AddV2Op>(loc, word_u64, increment);
      Value new_word = rewriter.create<CastOp>(loc, word_type, new_word_u64);
      pack_args.push_back(new_word);

      Value overflow = rewriter.create<LessOp>(loc, new_word_u64, word_u64);
      increment = rewriter.create<SelectV2Op>(loc, overflow, one_u64, zero_u64);
    }

    // Save the new state value to the resource.
    pack_args.push_back(key);
    Value new_state = rewriter.create<PackOp>(loc, res_type, pack_args);
    rewriter.create<AssignVariableOp>(loc, rng_op.resource(), new_state);

    // Pad the original state as necessary to fill the output shape.
    int pad = tensorflow::RNG_MAX_COUNTER_SIZE - counter_size;
    Type i64 = rewriter.getI64Type();
    RankedTensorType paddings_ty = RankedTensorType::get({1, 2}, i64);
    std::vector<int64_t> paddings_values = {0, pad};
    Value paddings = rewriter.create<ConstOp>(
        loc, DenseIntElementsAttr::get(paddings_ty, paddings_values));
    Value output = rewriter.create<PadOp>(loc, op_type, state, paddings);

    rewriter.replaceOp(op, output);
    return success();
  }
};

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_decompose_resource_ops.inc"
}  // namespace

void PopulateDecomposeResourceOpsPatterns(MLIRContext *context,
                                          RewritePatternSet *patterns) {
  patterns->add<DecomposeRngReadAndSkipOp>(context);
  populateWithGenerated(*patterns);
}

}  // namespace TF
}  // namespace mlir
