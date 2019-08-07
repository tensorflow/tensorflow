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

// This file defines the op traits used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_QUANTIZATION_TRAITS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_QUANTIZATION_TRAITS_H_

#include "mlir/Dialect/QuantOps/QuantTypes.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"

namespace mlir {
namespace OpTrait {
namespace TFL {

using QuantizedType = mlir::quant::QuantizedType;
using UniformQuantizedType = mlir::quant::UniformQuantizedType;

// The base class that all the quantization related OpTrait implements.
template <typename ConcreteType, template <typename> class TraitType>
struct QuantizationSpecTraitBase : public TraitBase<ConcreteType, TraitType> {
  static bool IsBias(int index) { return false; }
  static bool IsQuantizable() { return true; }
};

// This class provides the API for TFL ops that requires same input and output
// scale as the quantization results. This is used as a trait like this:
//
//   class TransposeOp
//       : public Op<TransposeOp, OpTrait::TFL::SameOperandsAndResultsScale> {
//
template <typename ConcreteType>
class SameOperandsAndResultsScale
    : public QuantizationSpecTraitBase<ConcreteType,
                                       SameOperandsAndResultsScale> {};

// This class provides the API for TFL ops that has a fixed output value range.
// This is used as a trait like this:
//
//   class SoftmaxOp
//       : public Op<SoftmaxOp,
//           OpTrait::TFL::FixedResultUniformScale<
//               8, -128, 390625, -8, 0, 255, false>::Impl> {
//
// TODO(fengliuai): create a better way to epxress floating point scale in the
// template argument list.
template <unsigned BitWidth, int ZeroPoint, int ScaleMantissa, int ScaleExp,
          int64_t StorageTypeMin, int64_t StorageTypeMax, bool Sign>
class FixedResultUniformScale {
 public:
  template <typename ConcreteType>
  class Impl
      : public QuantizationSpecTraitBase<
            ConcreteType, FixedResultUniformScale<
                              BitWidth, ZeroPoint, ScaleMantissa, ScaleExp,
                              StorageTypeMin, StorageTypeMax, Sign>::Impl> {
   public:
    QuantizedType GetResultQuantizedType(int index) {
      auto op = this->getOperation();
      auto result_type =
          op->getResult(index)->getType().template cast<TensorType>();
      Builder builder(op->getContext());
      IntegerType storage_type = builder.getIntegerType(BitWidth);
      const double scale = static_cast<double>(ScaleMantissa) *
                           ::pow(10.0, static_cast<double>(ScaleExp));
      return UniformQuantizedType::getChecked(
          Sign, storage_type, result_type.getElementType(), scale, ZeroPoint,
          StorageTypeMin, StorageTypeMax, builder.getUnknownLoc());
    }
  };
};

// This class provides the API for TFL ops that has input as bias. This is used
// as a trait like this:
//
//   class Conv2DOp
//       : public Op<Conv2DOp, OpTrait::TFL::AccumulatorScale<2, 0, 1>::Impl> {
//
// TODO(fengliuai): supports a configurable accumulator bit width.
template <int Bias, int... Operands>
class AccumulatorUniformScale {
 public:
  template <typename ConcreteType>
  class Impl
      : public QuantizationSpecTraitBase<
            ConcreteType, AccumulatorUniformScale<Bias, Operands...>::Impl> {
   public:
    // Whether the index-th operand is a bias.
    static bool IsBias(int index) { return index == Bias; }

    // Returns the indexes of all the non-bias operands.
    static std::vector<int> GetAllNonBiasOperands() {
      return std::vector<int>({Operands...});
    }
  };
};

// This class provides the API for TFL ops that shouldn't be quantized. This is
// used as a trait like this:
//
//   class LessOp : public Op<LessOp, OpTrait::TFL::NoQuantizableResult> {
//
template <typename ConcreteType>
class NoQuantizableResult
    : public QuantizationSpecTraitBase<ConcreteType, NoQuantizableResult> {
 public:
  static bool IsQuantizable() { return false; }
};

}  // namespace TFL
}  // namespace OpTrait
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_QUANTIZATION_TRAITS_H_
