/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_

#include <optional>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce,
          bool is_argmax>
class ConvertReduceOpToArgMinMax : public OpConversionPattern<mhlo::ReduceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ReduceOp reduce_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final;

  virtual bool IsValueInitValue(const DenseElementsAttr& attr) const = 0;
};

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMax
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        true> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   true>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};

// Base class for converting mhlo::ReduceOp to TF/TFL ArgMax/ArgMin ops.
template <typename Reduce, typename ArgReduce, typename BooleanReduce>
class ConvertReduceOpToArgMin
    : public ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                        false> {
 public:
  using ConvertReduceOpToArgMinMax<Reduce, ArgReduce, BooleanReduce,
                                   false>::ConvertReduceOpToArgMinMax;
  bool IsValueInitValue(const DenseElementsAttr& attr) const override;
};

using ConvertReduceOpToTFLiteArgmax =
    ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp, TFL::ReduceAnyOp>;
using ConvertReduceOpToTfArgmax =
    ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;
using ConvertReduceOpToTFLiteArgmin =
    ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp, TFL::ReduceAllOp>;
using ConvertReduceOpToTfArgmin =
    ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;

template class ConvertReduceOpToArgMax<TFL::ReduceMaxOp, TFL::ArgMaxOp,
                                       TFL::ReduceAnyOp>;
template class ConvertReduceOpToArgMin<TFL::ReduceMinOp, TFL::ArgMinOp,
                                       TFL::ReduceAllOp>;
template class ConvertReduceOpToArgMax<TF::MaxOp, TF::ArgMaxOp, TF::AnyOp>;
template class ConvertReduceOpToArgMin<TF::MinOp, TF::ArgMinOp, TF::AllOp>;

// Returns true if the given reduce op can be legalized to ArgMax/ArgMin ops.
std::optional<bool> IsReduceOpLegal(mhlo::ReduceOp reduce_op);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_LEGALIZE_HLO_CONVERSIONS_REDUCE_H_
