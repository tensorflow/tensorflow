/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <type_traits>  // NOLINT
#include <variant>      // NOLINT

#include "llvm/Support/ErrorHandling.h"
#include "tools/mlir_interpreter/dialects/comparators.h"
#include "tools/mlir_interpreter/dialects/cwise_math.h"
#include "tools/mlir_interpreter/framework/interpreter.h"
#include "tools/mlir_interpreter/framework/interpreter_value.h"
#include "tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "tools/mlir_interpreter/framework/registration.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue bitcast(InterpreterState&, arith::BitcastOp op,
                         const InterpreterValue& in) {
  Type ty = op->getResultTypes()[0];
  auto shapedTy = ty.dyn_cast<ShapedType>();
  auto result = dispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    TensorOrMemref<decltype(dummy)> result;
    result.view = {};
    if (shapedTy) {
      result.buffer = in.clone().buffer();
    } else {
      result.buffer = in.asUnitTensor().buffer();
    }
    return {result};
  });
  if (!shapedTy) {
    return result.extractElement({});
  }
  auto& outView = result.view();
  outView.strides = BufferView::getDefaultStrides(shapedTy.getShape());
  outView.sizes = llvm::to_vector(shapedTy.getShape());
  return result;
}

InterpreterValue constant(InterpreterState&, arith::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0];
  auto shapedType = ty.dyn_cast<ShapedType>();
  auto elemTy = shapedType ? shapedType.getElementType() : ty;
  return dispatchScalarType(elemTy, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    if (shapedType) {
      auto values =
          constant.getValue().cast<DenseElementsAttr>().getValues<T>();
      auto result = TensorOrMemref<T>::empty(shapedType.getShape());
      auto valueIt = values.begin();
      result.view.isVector = shapedType.isa<VectorType>();
      for (const auto& index : result.view.indices(true)) {
        result.at(index) = *valueIt;
        ++valueIt;
      }
      return {result};
    }

    auto value = constant.getValue();
    if (auto integer = value.dyn_cast<IntegerAttr>()) {
      return {static_cast<T>(integer.getInt())};
    }
    if (auto floatValue = value.dyn_cast<FloatAttr>()) {
      return {static_cast<T>(floatValue.getValueAsDouble())};
    }

    llvm_unreachable("unsupported constant type");
  });
}

template <typename Op>
InterpreterValue intCast(InterpreterState&, Op op,
                         const InterpreterValue& arg) {
  if (arg.isTensor()) {
    return dispatchScalarType(
        op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
          auto result = TensorOrMemref<decltype(dummy)>::emptyLike(arg.view());
          for (const auto& index : result.view.indices()) {
            result.at(index) =
                static_cast<decltype(dummy)>(arg.extractElement(index).asInt());
          }
          return {result};
        });
  }

  return dispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return {static_cast<decltype(dummy)>(arg.asInt())};
      });
}

llvm::SmallVector<InterpreterValue> uiToFP(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  if (args[0].isTensor()) {
    auto ty = op->getResultTypes()[0].cast<ShapedType>();
    return {dispatchScalarType(
        ty.getElementType(), [&](auto dummy) -> InterpreterValue {
          auto result =
              TensorOrMemref<decltype(dummy)>::emptyLike(args[0].view());
          for (const auto& index : result.view.indices()) {
            result.at(index) = static_cast<decltype(dummy)>(
                args[0].extractElement(index).asUInt());
          }
          return {result};
        })};
  }

  return {dispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return {static_cast<decltype(dummy)>(args[0].asUInt())};
      })};
}

InterpreterValue cmpI(InterpreterState&, arith::CmpIOp compare,
                      const InterpreterValue& lhs,
                      const InterpreterValue& rhs) {
  switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return applyCwiseBinaryMap<Foeq>(lhs, rhs);
    case arith::CmpIPredicate::ne:
      return applyCwiseBinaryMap<Fone>(lhs, rhs);
    case arith::CmpIPredicate::slt:
      return applyCwiseBinaryMap<Folt>(lhs, rhs);
    case arith::CmpIPredicate::sle:
      return applyCwiseBinaryMap<Fole>(lhs, rhs);
    case arith::CmpIPredicate::sgt:
      return applyCwiseBinaryMap<Fogt>(lhs, rhs);
    case arith::CmpIPredicate::sge:
      return applyCwiseBinaryMap<Foge>(lhs, rhs);
    case arith::CmpIPredicate::ult:
      return applyCwiseBinaryMap<Iult>(lhs, rhs);
    case arith::CmpIPredicate::ule:
      return applyCwiseBinaryMap<Iule>(lhs, rhs);
    case arith::CmpIPredicate::ugt:
      return applyCwiseBinaryMap<Iugt>(lhs, rhs);
    case arith::CmpIPredicate::uge:
      return applyCwiseBinaryMap<Iuge>(lhs, rhs);
  }
}

template <bool value>
struct ConstFunctor : CwiseAll {
  template <typename T>
  static bool apply(T, T) {
    return value;
  }
};

InterpreterValue cmpF(InterpreterState&, arith::CmpFOp compare,
                      const InterpreterValue& lhs,
                      const InterpreterValue& rhs) {
  switch (compare.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse:
      return applyCwiseBinaryMap<ConstFunctor<false>>(lhs, rhs);
    case arith::CmpFPredicate::OEQ:
      return applyCwiseBinaryMap<Foeq>(lhs, rhs);
    case arith::CmpFPredicate::OGT:
      return applyCwiseBinaryMap<Fogt>(lhs, rhs);
    case arith::CmpFPredicate::OGE:
      return applyCwiseBinaryMap<Foge>(lhs, rhs);
    case arith::CmpFPredicate::OLT:
      return applyCwiseBinaryMap<Folt>(lhs, rhs);
    case arith::CmpFPredicate::OLE:
      return applyCwiseBinaryMap<Fole>(lhs, rhs);
    case arith::CmpFPredicate::ONE:
      return applyCwiseBinaryMap<Fone>(lhs, rhs);
    case arith::CmpFPredicate::ORD:
      return applyCwiseBinaryMap<Ford>(lhs, rhs);
    case arith::CmpFPredicate::UEQ:
      return applyCwiseBinaryMap<Fueq>(lhs, rhs);
    case arith::CmpFPredicate::UGT:
      return applyCwiseBinaryMap<Fugt>(lhs, rhs);
    case arith::CmpFPredicate::UGE:
      return applyCwiseBinaryMap<Fuge>(lhs, rhs);
    case arith::CmpFPredicate::ULT:
      return applyCwiseBinaryMap<Fult>(lhs, rhs);
    case arith::CmpFPredicate::ULE:
      return applyCwiseBinaryMap<Fule>(lhs, rhs);
    case arith::CmpFPredicate::UNE:
      return applyCwiseBinaryMap<Fune>(lhs, rhs);
    case arith::CmpFPredicate::UNO:
      return applyCwiseBinaryMap<Funo>(lhs, rhs);
    case arith::CmpFPredicate::AlwaysTrue:
      return applyCwiseBinaryMap<ConstFunctor<true>>(lhs, rhs);
  }
}

InterpreterValue select(InterpreterState& state, arith::SelectOp,
                        const InterpreterValue& cond,
                        const InterpreterValue& trueValue,
                        const InterpreterValue& falseValue) {
  if (std::holds_alternative<bool>(cond.storage)) {
    return std::get<bool>(cond.storage) ? trueValue : falseValue;
  }

  if (!cond.isTensor() || !cond.view().isVector) {
    llvm::errs() << cond.toString();
    state.addFailure("select requires a scalar or vector argument");
    return {};
  }

  auto ret = trueValue.clone();
  for (const auto& index : cond.view().indices()) {
    if (cond.extractElement(index).asInt() == 0) {
      ret.insertElement(index, falseValue.extractElement(index));
    }
  }
  return ret;
}

template <typename R>
struct ExtFFunctor : CwiseFloat {
  template <typename A>
  static R apply(A v) {
    return v;
  }
};

InterpreterValue extF(InterpreterState&, arith::ExtFOp op,
                      const InterpreterValue& in) {
  return dispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return applyCwiseMap<ExtFFunctor<decltype(dummy)>>(in);
      });
}

REGISTER_MLIR_INTERPRETER_OP("arith.addf", applyCwiseBinaryMap<Plus>);
REGISTER_MLIR_INTERPRETER_OP("arith.andi", applyCwiseBinaryMap<BitAnd>);
REGISTER_MLIR_INTERPRETER_OP("arith.divf", applyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("arith.extui", uiToFP);
REGISTER_MLIR_INTERPRETER_OP("arith.maxf", applyCwiseBinaryMap<Max>);
REGISTER_MLIR_INTERPRETER_OP("arith.minf", applyCwiseBinaryMap<Min>);
REGISTER_MLIR_INTERPRETER_OP("arith.mulf", applyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("arith.negf", applyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("arith.ori", applyCwiseBinaryMap<BitOr>);
REGISTER_MLIR_INTERPRETER_OP("arith.remf", applyCwiseBinaryMap<Remainder>);
REGISTER_MLIR_INTERPRETER_OP("arith.subf", applyCwiseBinaryMap<Minus>);
REGISTER_MLIR_INTERPRETER_OP("arith.uitofp", uiToFP);
REGISTER_MLIR_INTERPRETER_OP("arith.xori", applyCwiseBinaryMap<BitXor>);
REGISTER_MLIR_INTERPRETER_OP("arith.shrui",
                             applyCwiseBinaryMap<ShiftRightLogical>);
REGISTER_MLIR_INTERPRETER_OP("arith.shli", applyCwiseBinaryMap<ShiftLeft>);

// The float implementations support ints too.
REGISTER_MLIR_INTERPRETER_OP("arith.addi", "arith.addf");
REGISTER_MLIR_INTERPRETER_OP("arith.divsi", "arith.divf");
REGISTER_MLIR_INTERPRETER_OP("arith.maxsi", "arith.maxf");
REGISTER_MLIR_INTERPRETER_OP("arith.minsi", "arith.minf");
REGISTER_MLIR_INTERPRETER_OP("arith.muli", "arith.mulf");
REGISTER_MLIR_INTERPRETER_OP("arith.subi", "arith.subf");

REGISTER_MLIR_INTERPRETER_OP(bitcast);
REGISTER_MLIR_INTERPRETER_OP(cmpF);
REGISTER_MLIR_INTERPRETER_OP(cmpI);
REGISTER_MLIR_INTERPRETER_OP(constant);
REGISTER_MLIR_INTERPRETER_OP(extF);
REGISTER_MLIR_INTERPRETER_OP(intCast<arith::IndexCastOp>);
REGISTER_MLIR_INTERPRETER_OP(intCast<arith::TruncIOp>);
REGISTER_MLIR_INTERPRETER_OP(intCast<arith::SIToFPOp>);
REGISTER_MLIR_INTERPRETER_OP(intCast<arith::ExtSIOp>);
REGISTER_MLIR_INTERPRETER_OP(select);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
