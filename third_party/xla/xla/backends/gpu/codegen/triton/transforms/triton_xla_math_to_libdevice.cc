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

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "xla/codegen/xtile/codegen/emitter_helpers.h"
#include "xla/service/gpu/target_util.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAMATHTOLIBDEVICEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

using ::xla::gpu::TargetDeviceFunctionID;

template <typename OpTy>
struct OpInfo;

template <>
struct OpInfo<mlir::math::AcosOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAcos;
};

template <>
struct OpInfo<mlir::math::AcoshOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAcosh;
};

template <>
struct OpInfo<mlir::math::AsinOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAsin;
};

template <>
struct OpInfo<mlir::math::AsinhOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAsinh;
};

template <>
struct OpInfo<mlir::math::Atan2Op> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAtan2;
};

template <>
struct OpInfo<mlir::math::AtanhOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kAtanh;
};

template <>
struct OpInfo<mlir::math::CosOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kCos;
};

template <>
struct OpInfo<mlir::math::CoshOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kCosh;
};

template <>
struct OpInfo<mlir::math::ExpOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kExp;
};

template <>
struct OpInfo<mlir::math::ErfOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kErf;
};

template <>
struct OpInfo<mlir::math::ExpM1Op> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kExpm1;
};

template <>
struct OpInfo<mlir::math::LogOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kLog;
};

template <>
struct OpInfo<mlir::math::Log1pOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kLog1p;
};

template <>
struct OpInfo<mlir::math::PowFOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kPow;
};

template <>
struct OpInfo<mlir::arith::RemFOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kFmod;
};

template <>
struct OpInfo<mlir::math::RsqrtOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kRsqrt;
};

template <>
struct OpInfo<mlir::math::SinOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kSin;
};

template <>
struct OpInfo<mlir::math::SinhOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kSinh;
};

template <>
struct OpInfo<mlir::math::SqrtOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kSqrt;
};

template <>
struct OpInfo<mlir::math::TanOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kTan;
};

template <>
struct OpInfo<mlir::math::TanhOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kTanh;
};

template <>
struct OpInfo<mlir::math::CbrtOp> {
  static constexpr auto kFunctionID = TargetDeviceFunctionID::kCbrt;
};

template <typename OpTy>
class ConvertToLibdevice : public mlir::OpRewritePattern<OpTy> {
 public:
  ConvertToLibdevice(mlir::MLIRContext* context,
                     absl::string_view libdevice_path,
                     const llvm::Triple& triple)
      : mlir::OpRewritePattern<OpTy>(context),
        libdevice_path_(libdevice_path),
        triple_(triple) {}

  mlir::LogicalResult matchAndRewrite(
      OpTy op, mlir::PatternRewriter& rewriter) const override {
    auto maybe_shaped_type = mlir::dyn_cast<mlir::ShapedType>(op.getType());
    mlir::Type output_type =
        maybe_shaped_type ? maybe_shaped_type.getElementType() : op.getType();

    bool output_type_is_16bit_float =
        output_type.isBF16() || output_type.isF16();
    if (!(output_type_is_16bit_float || output_type.isF32() ||
          output_type.isF64())) {
      op.emitError() << "unsupported output type";
      return rewriter.notifyMatchFailure(op, "unsupported output type");
    }

    absl::StatusOr<::xla::PrimitiveType> primitive_type_or =
        ::xla::xtile::GetPrimitiveType(output_type);
    if (!primitive_type_or.ok()) {
      return rewriter.notifyMatchFailure(op, "could not get primitive type");
    }

    mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

    llvm::SmallVector<Value, 2> casted_inputs;
    if (output_type_is_16bit_float) {
      // Upcast the inputs to F32.
      for (auto operand : op->getOperands()) {
        casted_inputs.push_back(
            ::xla::xtile::Cast(builder, operand, rewriter.getF32Type()));
      }
    } else {
      casted_inputs = llvm::to_vector(op->getOperands());
    }

    Value res = mlir::triton::ExternElementwiseOp::create(
        builder, casted_inputs[0].getType(), casted_inputs, "libdevice",
        libdevice_path_,
        ObtainDeviceFunctionName(OpInfo<OpTy>::kFunctionID, *primitive_type_or,
                                 triple_),
        /*pure=*/true);

    if (res.getType() != output_type) {
      // Downcast back to the original output type.
      res = ::xla::xtile::Cast(builder, res, output_type);
    }

    rewriter.replaceOp(op, res);

    return mlir::success();
  }

 private:
  // These are both owned by the parent pass (TritonXLAMathToLibdevicePass), so
  // it is safe to store references here.
  absl::string_view libdevice_path_;
  const llvm::Triple& triple_;
};

template <typename... OpTypes>
void AddPattens(mlir::RewritePatternSet& patterns,
                absl::string_view libdevice_path, const llvm::Triple& triple) {
  patterns.add<ConvertToLibdevice<OpTypes>...>(patterns.getContext(),
                                               libdevice_path, triple);
}

class TritonXLAMathToLibdevicePass
    : public impl::TritonXLAMathToLibdevicePassBase<
          TritonXLAMathToLibdevicePass> {
 public:
  using TritonXLAMathToLibdevicePassBase::TritonXLAMathToLibdevicePassBase;

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    mlir::RewritePatternSet patterns(context);

    llvm::Triple triple(triple_string_);

    AddPattens<mlir::math::AcosOp, mlir::math::AcoshOp, mlir::math::AsinOp,
               mlir::math::AsinhOp, mlir::math::Atan2Op, mlir::math::AtanhOp,
               mlir::math::CosOp, mlir::math::CoshOp, mlir::math::ExpOp,
               mlir::math::ErfOp, mlir::math::ExpM1Op, mlir::math::LogOp,
               mlir::math::Log1pOp, mlir::math::PowFOp, mlir::arith::RemFOp,
               mlir::math::RsqrtOp, mlir::math::SinOp, mlir::math::SinhOp,
               mlir::math::SqrtOp, mlir::math::TanOp, mlir::math::TanhOp,
               mlir::math::CbrtOp>(patterns, libdevice_path_, triple);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAMathToLibdevicePass(
    absl::string_view libdevice_path, absl::string_view triple) {
  TritonXLAMathToLibdevicePassOptions options;
  options.libdevice_path_ = libdevice_path;
  options.triple_string_ = triple;

  return std::make_unique<TritonXLAMathToLibdevicePass>(options);
}

}  // namespace mlir::triton::xla
