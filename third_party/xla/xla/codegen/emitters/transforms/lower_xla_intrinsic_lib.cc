/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"  // IWYU pragma: keep
#include "xla/codegen/emitters/utils.h"
#include "xla/codegen/intrinsic/cpp/intrinsic_declarations.h"
#include "xla/codegen/intrinsic/erf.h"
#include "xla/codegen/intrinsic/exp.h"
#include "xla/codegen/intrinsic/fptrunc.h"
#include "xla/codegen/intrinsic/log1p.h"
#include "xla/codegen/intrinsic/rsqrt.h"
#include "xla/codegen/intrinsic/tanh.h"
#include "xla/codegen/intrinsic/type.h"
#include "xla/mlir/utils/type_util.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERXLAINTRINSICLIBPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

using llvm::SmallVector;
using mlir::TypeRange;
using mlir::Value;

namespace mm = ::mlir::math;
namespace mv = ::mlir::vector;
namespace ma = ::mlir::arith;
namespace mf = ::mlir::func;
namespace ci = ::xla::codegen::intrinsics;

// TODO(talts): Add LowerMathOpPattern based on MathFunction instances.

// Get the vectorized version of the given element type if `vector_type` is
// present, else return the given element type.
mlir::Type GetVectorType(mlir::VectorType vector_type,
                         mlir::Type element_type) {
  return vector_type ? vector_type.clone(element_type) : element_type;
}

struct TargetTypes {
  SmallVector<ci::Type, 2> vector_types;
  SmallVector<ci::Type, 2> scalar_types;
  bool needs_upcast = false;
};

std::string GetCpuFeaturesStr(mlir::ModuleOp module_op) {
  mlir::StringAttr features =
      module_op->template getAttrOfType<mlir::StringAttr>("mhlo.cpu_features");
  return !features ? "" : features.getValue().str();
}

template <typename Intrinsic, typename Op>
TargetTypes GetTargetTypes(Op& op) {
  TargetTypes target_types;
  auto vector_type = mlir::dyn_cast<mlir::VectorType>(op.getType());
  for (mlir::Type type : TypeRange(op->getOperands())) {
    mlir::Type element_type = mlir::getElementTypeOrSelf(type);
    PrimitiveType target_type;
    if constexpr (!Intrinsic::kLastArgIsReturnType) {
      if (element_type.isBF16() || element_type.isF16()) {
        target_type = xla::F32;
        target_types.needs_upcast = true;
      } else {
        target_type = ConvertMlirTypeToPrimitiveType(element_type);
      }
    } else {
      target_type = ConvertMlirTypeToPrimitiveType(element_type);
    }
    if (vector_type) {
      target_types.vector_types.push_back(
          ci::Type::V(target_type, vector_type.getNumElements()));
    }
    target_types.scalar_types.push_back(ci::Type::S(target_type));
  }
  if constexpr (Intrinsic::kLastArgIsReturnType) {
    mlir::Type result_element_type = mlir::getElementTypeOrSelf(op.getType());
    PrimitiveType result_target_type =
        ConvertMlirTypeToPrimitiveType(result_element_type);
    if (vector_type) {
      target_types.vector_types.push_back(
          ci::Type::V(result_target_type, vector_type.getNumElements()));
    }
    target_types.scalar_types.push_back(ci::Type::S(result_target_type));
  }
  return target_types;
}

template <typename Intrinsic>
mlir::func::FuncOp GetDeclaration(mlir::ImplicitLocOpBuilder& b,
                                  mlir::ModuleOp module_op,
                                  llvm::ArrayRef<ci::Type> types) {
  return Intrinsic::GetOrInsertDeclaration(b, module_op, types);
}

template <typename Intrinsic, typename Op>
Value CallIntrinsicFunc(mlir::ImplicitLocOpBuilder& b, Op op,
                        const TargetTypes& target_types,
                        mlir::VectorType vec_type, bool vector_supported,
                        bool scalar_supported) {
  auto module_op = op->template getParentOfType<mlir::ModuleOp>();

  SmallVector<Value> compute_inputs;
  compute_inputs.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    mlir::Type elem_type = mlir::getElementTypeOrSelf(operand.getType());
    if (target_types.needs_upcast &&
        (elem_type.isF16() || elem_type.isBF16())) {
      compute_inputs.push_back(
          EmitFloatCast(operand, GetVectorType(vec_type, b.getF32Type()), b));
    } else {
      compute_inputs.push_back(operand);
    }
  }
  if (vector_supported) {
    auto intrinsic_decl =
        GetDeclaration<Intrinsic>(b, module_op, target_types.vector_types);
    Value compute_result =
        mf::CallOp::create(b, intrinsic_decl, compute_inputs).getResult(0);
    return target_types.needs_upcast
               ? EmitFloatCast(compute_result, op.getType(), b)
               : compute_result;
  }

  // Fallback to scalar.
  auto intrinsic_decl =
      GetDeclaration<Intrinsic>(b, module_op, target_types.scalar_types);

  if (!vec_type) {
    Value scalar_result =
        mf::CallOp::create(b, intrinsic_decl, compute_inputs).getResult(0);
    return target_types.needs_upcast
               ? ma::TruncFOp::create(b, op.getType(), scalar_result)
               : scalar_result;
  }

  llvm::SmallVector<Value> scalar_results;
  scalar_results.reserve(vec_type.getNumElements());
  for (int64_t idx = 0; idx != vec_type.getNumElements(); ++idx) {
    SmallVector<Value> scalar_inputs;
    scalar_inputs.reserve(compute_inputs.size());
    for (Value compute_input : compute_inputs) {
      if (mlir::isa<mlir::VectorType>(compute_input.getType())) {
        scalar_inputs.push_back(mv::ExtractOp::create(b, compute_input, idx));
      } else {
        scalar_inputs.push_back(compute_input);
      }
    }
    Value scalar_result =
        mf::CallOp::create(b, intrinsic_decl, scalar_inputs).getResult(0);
    scalar_results.push_back(scalar_result);
  }
  mlir::Type compute_type = target_types.needs_upcast
                                ? GetVectorType(vec_type, b.getF32Type())
                                : op.getType();
  Value compute_result =
      mv::FromElementsOp::create(b, compute_type, scalar_results);

  return target_types.needs_upcast
             ? ma::TruncFOp::create(b, op.getType(), compute_result)
             : compute_result;
}

template <typename Intrinsic, typename Op>
mlir::LogicalResult LowerIntrinsicPattern(Op op,
                                          mlir::PatternRewriter& rewriter) {
  auto vec_type = mlir::dyn_cast<mlir::VectorType>(op.getType());
  if (vec_type && vec_type.getRank() != 1) {
    // These will later be converted to loops of 1D vectors but will then miss
    // the XLA intrinsic lowering.
    op->emitWarning() << "Missed XLA intrinsic lowering as vector rank != 1.";
    return rewriter.notifyMatchFailure(op, "Vector rank is not 1.");
  }

  mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  TargetTypes target_types = GetTargetTypes<Intrinsic>(op);
  auto module_op = op->template getParentOfType<mlir::ModuleOp>();

  std::string features_str = GetCpuFeaturesStr(module_op);
  bool vector_supported =
      Intrinsic::IsSupported(features_str, target_types.vector_types);
  bool scalar_supported =
      Intrinsic::IsSupported(features_str, target_types.scalar_types);
  if (!vector_supported && !scalar_supported) {
    return rewriter.notifyMatchFailure(op, "unsupported type");
  }
  Value result = CallIntrinsicFunc<Intrinsic>(
      b, op, target_types, vec_type, vector_supported, scalar_supported);
  rewriter.replaceOp(op, result);
  return mlir::success();
}

class LowerXlaIntrinsicLibPass
    : public impl::LowerXlaIntrinsicLibPassBase<LowerXlaIntrinsicLibPass> {
 public:
  LowerXlaIntrinsicLibPass()
      : impl::LowerXlaIntrinsicLibPassBase<LowerXlaIntrinsicLibPass>() {}

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::ModuleOp module_op = getOperation();
    mlir::RewritePatternSet patterns(context);
    patterns.add(LowerIntrinsicPattern<ci::Exp, mm::ExpOp>);
    patterns.add(LowerIntrinsicPattern<ci::Log1p, mm::Log1pOp>);
    patterns.add(LowerIntrinsicPattern<ci::Rsqrt, mm::RsqrtOp>);
    patterns.add(LowerIntrinsicPattern<ci::Tanh, mm::TanhOp>);
    patterns.add(LowerIntrinsicPattern<ci::EigenAtan, mm::AtanOp>);
    patterns.add(LowerIntrinsicPattern<ci::EigenSin, mm::SinOp>);
    patterns.add(LowerIntrinsicPattern<ci::EigenCos, mm::CosOp>);
    patterns.add(LowerIntrinsicPattern<ci::FpTrunc, ma::TruncFOp>);
    patterns.add(LowerIntrinsicPattern<ci::Erf, mm::ErfOp>);
    if (mlir::failed(
            mlir::applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace emitters
}  // namespace xla
