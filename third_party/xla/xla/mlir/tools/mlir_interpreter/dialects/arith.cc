/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include <variant>  // NOLINT

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/comparators.h"
#include "xla/mlir/tools/mlir_interpreter/dialects/cwise_math.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value.h"
#include "xla/mlir/tools/mlir_interpreter/framework/interpreter_value_util.h"
#include "xla/mlir/tools/mlir_interpreter/framework/registration.h"
#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

namespace mlir {
namespace interpreter {
namespace {

InterpreterValue Bitcast(InterpreterState&, arith::BitcastOp op,
                         const InterpreterValue& in) {
  Type ty = op->getResultTypes()[0];
  auto shaped_ty = dyn_cast<ShapedType>(ty);
  auto result = DispatchScalarType(ty, [&](auto dummy) -> InterpreterValue {
    TensorOrMemref<decltype(dummy)> result;
    result.view = {};
    if (shaped_ty) {
      result.buffer = in.Clone().GetBuffer();
    } else {
      result.buffer = in.AsUnitTensor().GetBuffer();
    }
    return {result};
  });
  if (!shaped_ty) {
    return result.ExtractElement({});
  }
  auto& out_view = result.View();
  out_view.strides = BufferView::GetDefaultStrides(shaped_ty.getShape());
  out_view.sizes = llvm::to_vector(shaped_ty.getShape());
  return result;
}

InterpreterValue Constant(InterpreterState&, arith::ConstantOp constant) {
  auto ty = constant->getResultTypes()[0];
  auto shaped_ty = dyn_cast<ShapedType>(ty);
  auto elem_ty = shaped_ty ? shaped_ty.getElementType() : ty;
  return DispatchScalarType(elem_ty, [&](auto dummy) -> InterpreterValue {
    using T = decltype(dummy);
    if (shaped_ty) {
      auto values = cast<DenseElementsAttr>(constant.getValue()).getValues<T>();
      auto result = TensorOrMemref<T>::Empty(shaped_ty.getShape());
      auto value_it = values.begin();
      result.view.is_vector = isa<VectorType>(shaped_ty);
      for (const auto& index : result.view.Indices(true)) {
        result.at(index) = *value_it;
        ++value_it;
      }
      return {result};
    }

    auto value = constant.getValue();
    if (auto integer = mlir::dyn_cast<IntegerAttr>(value)) {
      return {static_cast<T>(integer.getInt())};
    }
    if (auto float_value = mlir::dyn_cast<FloatAttr>(value)) {
      return {static_cast<T>(float_value.getValueAsDouble())};
    }

    llvm_unreachable("unsupported constant type");
  });
}

template <typename Op>
InterpreterValue IntCast(InterpreterState&, Op op,
                         const InterpreterValue& arg) {
  if (arg.IsTensor()) {
    return DispatchScalarType(
        op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
          auto result = TensorOrMemref<decltype(dummy)>::EmptyLike(arg.View());
          for (const auto& index : result.view.Indices()) {
            result.at(index) =
                static_cast<decltype(dummy)>(arg.ExtractElement(index).AsInt());
          }
          return {result};
        });
  }

  return DispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return {static_cast<decltype(dummy)>(arg.AsInt())};
      });
}

template <typename Op>
InterpreterValue FloatCast(InterpreterState&, Op op,
                           const InterpreterValue& arg) {
  if (arg.IsTensor()) {
    return DispatchScalarType(
        op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
          auto result = TensorOrMemref<decltype(dummy)>::EmptyLike(arg.View());
          for (const auto& index : result.view.Indices()) {
            result.at(index) = static_cast<decltype(dummy)>(
                arg.ExtractElement(index).AsDouble());
          }
          return {result};
        });
  }

  return DispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return {static_cast<decltype(dummy)>(arg.AsDouble())};
      });
}

llvm::SmallVector<InterpreterValue> UiToFP(
    MutableArrayRef<InterpreterValue> args, mlir::Operation* op,
    InterpreterState&) {
  if (args[0].IsTensor()) {
    auto ty = mlir::cast<ShapedType>(op->getResultTypes()[0]);
    return {DispatchScalarType(
        ty.getElementType(), [&](auto dummy) -> InterpreterValue {
          auto result =
              TensorOrMemref<decltype(dummy)>::EmptyLike(args[0].View());
          for (const auto& index : result.view.Indices()) {
            result.at(index) = static_cast<decltype(dummy)>(
                args[0].ExtractElement(index).AsUInt());
          }
          return {result};
        })};
  }

  return {DispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return {static_cast<decltype(dummy)>(args[0].AsUInt())};
      })};
}

InterpreterValue CmpI(InterpreterState&, arith::CmpIOp compare,
                      const InterpreterValue& lhs,
                      const InterpreterValue& rhs) {
  switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return ApplyCwiseBinaryMap<Foeq>(lhs, rhs);
    case arith::CmpIPredicate::ne:
      return ApplyCwiseBinaryMap<Fone>(lhs, rhs);
    case arith::CmpIPredicate::slt:
      return ApplyCwiseBinaryMap<Folt>(lhs, rhs);
    case arith::CmpIPredicate::sle:
      return ApplyCwiseBinaryMap<Fole>(lhs, rhs);
    case arith::CmpIPredicate::sgt:
      return ApplyCwiseBinaryMap<Fogt>(lhs, rhs);
    case arith::CmpIPredicate::sge:
      return ApplyCwiseBinaryMap<Foge>(lhs, rhs);
    case arith::CmpIPredicate::ult:
      return ApplyCwiseBinaryMap<Iult>(lhs, rhs);
    case arith::CmpIPredicate::ule:
      return ApplyCwiseBinaryMap<Iule>(lhs, rhs);
    case arith::CmpIPredicate::ugt:
      return ApplyCwiseBinaryMap<Iugt>(lhs, rhs);
    case arith::CmpIPredicate::uge:
      return ApplyCwiseBinaryMap<Iuge>(lhs, rhs);
  }
}

template <bool value>
struct ConstFunctor : CwiseAll {
  template <typename T>
  static bool Apply(T, T) {
    return value;
  }
};

InterpreterValue CmpF(InterpreterState&, arith::CmpFOp compare,
                      const InterpreterValue& lhs,
                      const InterpreterValue& rhs) {
  switch (compare.getPredicate()) {
    case arith::CmpFPredicate::AlwaysFalse:
      return ApplyCwiseBinaryMap<ConstFunctor<false>>(lhs, rhs);
    case arith::CmpFPredicate::OEQ:
      return ApplyCwiseBinaryMap<Foeq>(lhs, rhs);
    case arith::CmpFPredicate::OGT:
      return ApplyCwiseBinaryMap<Fogt>(lhs, rhs);
    case arith::CmpFPredicate::OGE:
      return ApplyCwiseBinaryMap<Foge>(lhs, rhs);
    case arith::CmpFPredicate::OLT:
      return ApplyCwiseBinaryMap<Folt>(lhs, rhs);
    case arith::CmpFPredicate::OLE:
      return ApplyCwiseBinaryMap<Fole>(lhs, rhs);
    case arith::CmpFPredicate::ONE:
      return ApplyCwiseBinaryMap<Fone>(lhs, rhs);
    case arith::CmpFPredicate::ORD:
      return ApplyCwiseBinaryMap<Ford>(lhs, rhs);
    case arith::CmpFPredicate::UEQ:
      return ApplyCwiseBinaryMap<Fueq>(lhs, rhs);
    case arith::CmpFPredicate::UGT:
      return ApplyCwiseBinaryMap<Fugt>(lhs, rhs);
    case arith::CmpFPredicate::UGE:
      return ApplyCwiseBinaryMap<Fuge>(lhs, rhs);
    case arith::CmpFPredicate::ULT:
      return ApplyCwiseBinaryMap<Fult>(lhs, rhs);
    case arith::CmpFPredicate::ULE:
      return ApplyCwiseBinaryMap<Fule>(lhs, rhs);
    case arith::CmpFPredicate::UNE:
      return ApplyCwiseBinaryMap<Fune>(lhs, rhs);
    case arith::CmpFPredicate::UNO:
      return ApplyCwiseBinaryMap<Funo>(lhs, rhs);
    case arith::CmpFPredicate::AlwaysTrue:
      return ApplyCwiseBinaryMap<ConstFunctor<true>>(lhs, rhs);
  }
}

InterpreterValue Select(InterpreterState& state, arith::SelectOp,
                        const InterpreterValue& cond,
                        const InterpreterValue& true_value,
                        const InterpreterValue& false_value) {
  if (std::holds_alternative<bool>(cond.storage)) {
    return std::get<bool>(cond.storage) ? true_value : false_value;
  }

  if (!cond.IsTensor() && !cond.View().is_vector) {
    state.AddFailure("select requires a scalar or vector argument");
    return {};
  }

  auto ret = true_value.Clone();
  for (const auto& index : cond.View().Indices(/*include_vector_dims=*/true)) {
    if (cond.ExtractElement(index).AsInt() == 0) {
      ret.InsertElement(index, false_value.ExtractElement(index));
    }
  }
  return ret;
}

template <typename R>
struct ExtFFunctor : CwiseFloat {
  template <typename A>
  static R Apply(A v) {
    return v;
  }
};

InterpreterValue ExtF(InterpreterState&, arith::ExtFOp op,
                      const InterpreterValue& in) {
  return DispatchScalarType(
      op->getResultTypes()[0], [&](auto dummy) -> InterpreterValue {
        return ApplyCwiseMap<ExtFFunctor<decltype(dummy)>>(in);
      });
}

REGISTER_MLIR_INTERPRETER_OP("arith.addf", ApplyCwiseBinaryMap<Plus>);
REGISTER_MLIR_INTERPRETER_OP("arith.andi", ApplyCwiseBinaryMap<BitAnd>);
REGISTER_MLIR_INTERPRETER_OP("arith.divf", ApplyCwiseBinaryMap<Divide>);
REGISTER_MLIR_INTERPRETER_OP("arith.extui", UiToFP);
REGISTER_MLIR_INTERPRETER_OP("arith.maxf", ApplyCwiseBinaryMap<Max>);
REGISTER_MLIR_INTERPRETER_OP("arith.minf", ApplyCwiseBinaryMap<Min>);
REGISTER_MLIR_INTERPRETER_OP("arith.mulf", ApplyCwiseBinaryMap<Multiply>);
REGISTER_MLIR_INTERPRETER_OP("arith.negf", ApplyCwiseMap<Neg>);
REGISTER_MLIR_INTERPRETER_OP("arith.ori", ApplyCwiseBinaryMap<BitOr>);
REGISTER_MLIR_INTERPRETER_OP("arith.remf", ApplyCwiseBinaryMap<Remainder>);
REGISTER_MLIR_INTERPRETER_OP("arith.subf", ApplyCwiseBinaryMap<Minus>);
REGISTER_MLIR_INTERPRETER_OP("arith.uitofp", UiToFP);
REGISTER_MLIR_INTERPRETER_OP("arith.xori", ApplyCwiseBinaryMap<BitXor>);
REGISTER_MLIR_INTERPRETER_OP("arith.shrui",
                             ApplyCwiseBinaryMap<ShiftRightLogical>);
REGISTER_MLIR_INTERPRETER_OP("arith.shrsi",
                             ApplyCwiseBinaryMap<ShiftRightArith>);
REGISTER_MLIR_INTERPRETER_OP("arith.shli", ApplyCwiseBinaryMap<ShiftLeft>);

// The float implementations support ints too.
REGISTER_MLIR_INTERPRETER_OP("arith.addi", "arith.addf");
REGISTER_MLIR_INTERPRETER_OP("arith.divsi", "arith.divf");
REGISTER_MLIR_INTERPRETER_OP("arith.maxsi", "arith.maxf");
REGISTER_MLIR_INTERPRETER_OP("arith.minsi", "arith.minf");
REGISTER_MLIR_INTERPRETER_OP("arith.muli", "arith.mulf");
REGISTER_MLIR_INTERPRETER_OP("arith.remsi", "arith.remf");
REGISTER_MLIR_INTERPRETER_OP("arith.subi", "arith.subf");

REGISTER_MLIR_INTERPRETER_OP(Bitcast);
REGISTER_MLIR_INTERPRETER_OP(CmpF);
REGISTER_MLIR_INTERPRETER_OP(CmpI);
REGISTER_MLIR_INTERPRETER_OP(Constant);
REGISTER_MLIR_INTERPRETER_OP(ExtF);
REGISTER_MLIR_INTERPRETER_OP(FloatCast<arith::FPToSIOp>);
REGISTER_MLIR_INTERPRETER_OP(IntCast<arith::ExtSIOp>);
REGISTER_MLIR_INTERPRETER_OP(IntCast<arith::IndexCastOp>);
REGISTER_MLIR_INTERPRETER_OP(IntCast<arith::SIToFPOp>);
REGISTER_MLIR_INTERPRETER_OP(IntCast<arith::TruncIOp>);
REGISTER_MLIR_INTERPRETER_OP(Select);

}  // namespace
}  // namespace interpreter
}  // namespace mlir
