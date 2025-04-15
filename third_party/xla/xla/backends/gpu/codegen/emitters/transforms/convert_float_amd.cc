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

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTFLOATAMDPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace LLVM = ::mlir::LLVM;
namespace arith = ::mlir::arith;
namespace vector = ::mlir::vector;

template <typename SourceOp>
struct Fp8OpRewritePattern : public mlir::OpRewritePattern<SourceOp> {
  using FixedVectorValue = mlir::TypedValue<mlir::FixedVectorType>;
  using FloatValue = mlir::TypedValue<mlir::FloatType>;
  Fp8OpRewritePattern(mlir::MLIRContext* context, bool nativeNanooFp8)
      : mlir::OpRewritePattern<SourceOp>(context),
        nativeNanooFp8_(nativeNanooFp8) {}
  bool isFp8(const mlir::Type& type) const {
    return nativeNanooFp8_ ? llvm::isa<mlir::Float8E4M3FNUZType>(type)
                           : llvm::isa<mlir::Float8E4M3FNType>(type);
  }
  bool isBf8(const mlir::Type& type) const {
    return nativeNanooFp8_ ? llvm::isa<mlir::Float8E5M2FNUZType>(type)
                           : llvm::isa<mlir::Float8E5M2Type>(type);
  }

 private:
  bool nativeNanooFp8_;
};

struct RewriteFp8TruncFPattern : public Fp8OpRewritePattern<arith::TruncFOp> {
  using Fp8OpRewritePattern::Fp8OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      arith::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!isFp8(dst_ty) && !isBf8(dst_ty)) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    auto match = MatchBuildVector(op, src, dst_ty);

    if (match) {
      auto [inputs, output] = *match;
      rewriter.setInsertionPointAfter(output.getDefiningOp());
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      rewriter.replaceOp(
          output.getDefiningOp(),
          EmitVectorizedTruncToF8Intrinsic(inputs, output.getType(), b));
      return mlir::success();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitTruncToF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }

  std::optional<std::tuple<llvm::SmallVector<mlir::Value, 4>, FixedVectorValue>>
  MatchBuildVector(arith::TruncFOp op, FloatValue value,
                   mlir::FloatType to_ty) const {
    auto matchPos = [](vector::InsertOp insert, size_t* pos) -> bool {
      llvm::APInt ap_pos;
      auto position = insert.getMixedPosition();
      if (position.size() != 1) {
        return false;
      }
      if (auto attr = mlir::dyn_cast<mlir::Attribute>(position[0])) {
        if (!mlir::matchPattern(attr, mlir::m_ConstantInt(&ap_pos))) {
          return false;
        }
      } else {
        if (!mlir::matchPattern(mlir::cast<mlir::Value>(position[0]),
                                mlir::m_ConstantInt(&ap_pos))) {
          return false;
        }
      }

      *pos = ap_pos.getZExtValue();
      return true;
    };

    if (!op->hasOneUse()) {
      return std::nullopt;
    }

    size_t pos;
    auto insert = mlir::dyn_cast<vector::InsertOp>(op->use_begin()->getOwner());
    if (!insert || insert.getValueToStore() != op->getResult(0) ||
        !matchPos(insert, &pos) || !insert.getDest().hasOneUse()) {
      return std::nullopt;
    }

    mlir::Value vector = insert.getDest();

    size_t element_count =
        mlir::cast<FixedVectorValue>(vector).getType().getNumElements();

    if (!llvm::isPowerOf2_64(element_count) || element_count == 1) {
      return std::nullopt;
    }

    llvm::SmallVector<mlir::Value, 4> inputs(element_count);

    auto addInput = [&](mlir::Value input, size_t index) -> bool {
      if (index >= element_count) {
        return false;
      }
      if (inputs[index]) {
        return false;
      }
      inputs[index] = input;
      return true;
    };

    addInput(value, pos);

    mlir::Value input;
    mlir::Operation* to_match = vector.getDefiningOp();
    while (mlir::matchPattern(to_match, mlir::m_Op<vector::InsertOp>(
                                            mlir::m_Op<arith::TruncFOp>(
                                                mlir::matchers::m_Any(&input)),
                                            mlir::matchers::m_Any(&vector))) &&
           matchPos(mlir::cast<vector::InsertOp>(to_match), &pos) &&
           vector.hasOneUse()) {
      if (!addInput(input, pos)) {
        return std::nullopt;
      }
      to_match = vector.getDefiningOp();
    }

    while (
        insert->hasOneUse() &&
        mlir::matchPattern(
            insert->use_begin()->getOwner(),
            mlir::m_Op<vector::InsertOp>(
                mlir::m_Op<arith::TruncFOp>(mlir::matchers::m_Any(&input)),
                mlir::matchers::m_Val(insert->getResult(0)))) &&
        matchPos(mlir::cast<vector::InsertOp>(insert->use_begin()->getOwner()),
                 &pos) &&
        input.getType() == value.getType()) {
      if (!addInput(input, pos)) {
        return std::nullopt;
      }
      insert = mlir::cast<vector::InsertOp>(insert->use_begin()->getOwner());
    }

    if (llvm::any_of(inputs, [](mlir::Value input) { return !input; })) {
      return std::nullopt;
    }
    return std::make_tuple(std::move(inputs),
                           mlir::cast<FixedVectorValue>(insert->getResult(0)));
  }

  mlir::Value EmitVectorizedTruncToF8Intrinsic(
      llvm::SmallVector<mlir::Value, 4>& inputs, mlir::FixedVectorType to_ty,
      mlir::ImplicitLocOpBuilder& b) const {
    assert(isFp8(to_ty.getElementType()) || isBf8(to_ty.getElementType()));

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    mlir::IntegerType i8_ty = b.getI8Type();
    mlir::IntegerType i1_ty = b.getI1Type();

    llvm::transform(inputs, inputs.begin(), [&](mlir::Value v) -> mlir::Value {
      if (v.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
        return b.create<arith::ExtFOp>(f32_ty, v);
      } else if (v.getType() != f32_ty) {
        return b.create<arith::TruncFOp>(f32_ty, v);
      } else {
        return v;
      }
    });

    mlir::StringAttr cvtIntr = b.getStringAttr(
        isFp8(to_ty.getElementType()) ? "llvm.amdgcn.cvt.pk.fp8.f32"
                                      : "llvm.amdgcn.cvt.pk.bf8.f32");

    size_t num_elements = to_ty.getNumElements();
    assert(num_elements == inputs.size() &&
           (num_elements == 2 || num_elements % 4 == 0));

    size_t num_chunks = (num_elements + 2) / 4;

    mlir::Type chunks_ty = LLVM::getFixedVectorType(i32_ty, num_chunks);
    mlir::Value chunks = b.create<LLVM::UndefOp>(chunks_ty);
    bool pos = false;
    for (size_t i = 0; i < inputs.size() / 2; i++) {
      mlir::Value chunk_pos = b.create<LLVM::ConstantOp>(i32_ty, 2 * i / 4);
      mlir::Value chunk = b.create<LLVM::ExtractElementOp>(chunks, chunk_pos);
      LLVM::CallIntrinsicOp cvtOp = b.create<LLVM::CallIntrinsicOp>(
          i32_ty, cvtIntr,
          mlir::ValueRange{inputs[2 * i], inputs[2 * i + 1], chunk,
                           b.create<LLVM::ConstantOp>(i1_ty, pos)});
      chunks = b.create<LLVM::InsertElementOp>(chunks, cvtOp.getResult(0),
                                               chunk_pos);
      pos ^= true;
    }

    if (num_elements == 2) {
      return b
          .create<mlir::UnrealizedConversionCastOp>(
              to_ty,
              mlir::ValueRange{b.create<LLVM::BitcastOp>(
                  LLVM::getFixedVectorType(i8_ty, num_elements),
                  b.create<LLVM::ExtractElementOp>(
                      b.create<LLVM::BitcastOp>(
                          LLVM::getFixedVectorType(b.getI16Type(), 2), chunks),
                      b.create<LLVM::ConstantOp>(i32_ty, 0)))})
          .getResult(0);
    }

    return b
        .create<mlir::UnrealizedConversionCastOp>(
            to_ty, mlir::ValueRange{b.create<LLVM::BitcastOp>(
                       LLVM::getFixedVectorType(i8_ty, num_elements), chunks)})
        .getResult(0);
  }

  mlir::Value EmitTruncToF8Intrinsic(mlir::Value value, mlir::FloatType to_ty,
                                     mlir::ImplicitLocOpBuilder& b) const {
    assert(isFp8(to_ty) || isBf8(to_ty));

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    if (value.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
      value = b.create<arith::ExtFOp>(f32_ty, value);
    } else if (value.getType() != f32_ty) {
      value = b.create<arith::TruncFOp>(f32_ty, value);
    }

    mlir::StringAttr cvtIntr =
        b.getStringAttr(isFp8(to_ty) ? "llvm.amdgcn.cvt.pk.fp8.f32"
                                     : "llvm.amdgcn.cvt.pk.bf8.f32");

    LLVM::CallIntrinsicOp cvtOp = b.create<LLVM::CallIntrinsicOp>(
        i32_ty, cvtIntr,
        mlir::ValueRange{value, b.create<LLVM::UndefOp>(f32_ty),
                         b.create<LLVM::UndefOp>(i32_ty),
                         b.create<LLVM::ConstantOp>(b.getI1Type(), 0)});
    mlir::Value res =
        b.create<LLVM::TruncOp>(b.getI8Type(), cvtOp.getResults());
    return b
        .create<mlir::UnrealizedConversionCastOp>(to_ty, mlir::ValueRange{res})
        .getResult(0);
  }
};

struct RewriteFp8ExtFPattern : public Fp8OpRewritePattern<arith::ExtFOp> {
  using Fp8OpRewritePattern::Fp8OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      arith::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!isFp8(src.getType()) && !isBf8(src.getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    auto match = MatchDecomposeVector(op, src, dst_ty);

    if (match) {
      auto [input, outputs] = *match;
      if (mlir::Operation* input_op = input.getDefiningOp()) {
        rewriter.setInsertionPointAfter(input_op);
      } else {
        rewriter.setInsertionPointToStart(
            mlir::cast<mlir::BlockArgument>(input).getOwner());
      }
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      auto new_outputs = EmitVectorizedExtFromF8Intrinsic(
          input, mlir::cast<mlir::FloatType>(outputs[0].getType()), b);
      for (auto [old_value, new_value] :
           llvm::zip_equal(outputs, new_outputs)) {
        rewriter.replaceOp(old_value.getDefiningOp(), new_value);
      }

      return mlir::success();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitExtFromF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }
  std::optional<std::tuple<FixedVectorValue, llvm::SmallVector<mlir::Value, 4>>>
  MatchDecomposeVector(arith::ExtFOp op, FloatValue value,
                       mlir::FloatType to_ty) const {
    auto matchPos = [](vector::ExtractOp extract, size_t* pos) -> bool {
      llvm::APInt ap_pos;
      auto position = extract.getMixedPosition();
      if (position.size() != 1) {
        return false;
      }
      if (auto attr = mlir::dyn_cast<mlir::Attribute>(position[0])) {
        if (!mlir::matchPattern(attr, mlir::m_ConstantInt(&ap_pos))) {
          return false;
        }
      } else {
        if (!mlir::matchPattern(mlir::cast<mlir::Value>(position[0]),
                                mlir::m_ConstantInt(&ap_pos))) {
          return false;
        }
      }
      *pos = ap_pos.getZExtValue();
      return true;
    };

    size_t pos;
    auto extract = value.getDefiningOp<vector::ExtractOp>();
    if (!extract || !extract->hasOneUse() || !matchPos(extract, &pos)) {
      return std::nullopt;
    }

    mlir::Value vector = extract.getVector();

    size_t element_count =
        mlir::cast<FixedVectorValue>(vector).getType().getNumElements();

    if (!llvm::isPowerOf2_64(element_count) || element_count == 1) {
      return std::nullopt;
    }

    llvm::SmallVector<mlir::Value, 4> outputs(element_count);

    auto addOutput = [&](mlir::Value output, size_t index) -> bool {
      if (index >= element_count) {
        return false;
      }
      if (outputs[index]) {
        return false;
      }
      outputs[index] = output;
      return true;
    };

    for (const mlir::OpOperand& use : vector.getUses()) {
      extract = mlir::dyn_cast<vector::ExtractOp>(use.getOwner());
      if (!extract || !extract->hasOneUse() || extract.getVector() != vector ||
          !matchPos(extract, &pos)) {
        return std::nullopt;
      }
      auto extf =
          mlir::dyn_cast<arith::ExtFOp>(extract->use_begin()->getOwner());
      if (!extf || extf.getType() != to_ty || extf.getOperand() != extract) {
        return std::nullopt;
      }
      if (!addOutput(extf, pos)) {
        return std::nullopt;
      }
    }

    if (llvm::any_of(outputs, [](mlir::Value output) { return !output; })) {
      return std::nullopt;
    }
    return std::make_tuple(mlir::cast<FixedVectorValue>(vector),
                           std::move(outputs));
  }

  mlir::Value ConvertFromFloat(mlir::Value v, mlir::FloatType to_ty,
                               mlir::ImplicitLocOpBuilder& b) const {
    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    if (to_ty == f32_ty) {
      return v;
    }

    if (to_ty.getWidth() > f32_ty.getWidth()) {
      return b.create<arith::ExtFOp>(to_ty, v);
    }

    if (to_ty.isBF16()) {
      return b.create<LLVM::BitcastOp>(
          to_ty,
          b.create<LLVM::TruncOp>(
              b.getI16Type(),
              b.create<LLVM::LShrOp>(b.create<LLVM::BitcastOp>(i32_ty, v),
                                     b.create<LLVM::ConstantOp>(i32_ty, 16))));
    }

    assert(to_ty.getWidth() < f32_ty.getWidth());
    return b.create<arith::TruncFOp>(to_ty, v);
  }

  llvm::SmallVector<mlir::Value, 4> EmitVectorizedExtFromF8Intrinsic(
      FixedVectorValue value, mlir::FloatType to_ty,
      mlir::ImplicitLocOpBuilder& b) const {
    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    mlir::IntegerType i16_ty = b.getI16Type();
    mlir::IntegerType i8_ty = b.getI8Type();
    mlir::IntegerType i1_ty = b.getI1Type();
    mlir::Value zero_cst = b.create<LLVM::ConstantOp>(i32_ty, 0);
    mlir::Value one_cst = b.create<LLVM::ConstantOp>(i32_ty, 1);

    size_t num_elements = value.getType().getNumElements();
    assert(num_elements == 2 || num_elements % 4 == 0);

    size_t num_chunks = (num_elements + 2) / 4;
    mlir::Type chunks_ty = LLVM::getFixedVectorType(i32_ty, num_chunks);
    mlir::Value chunks;

    if (num_elements == 2) {
      chunks = b.create<LLVM::BitcastOp>(
          chunks_ty,
          b.create<LLVM::InsertElementOp>(
              b.create<LLVM::UndefOp>(LLVM::getFixedVectorType(i16_ty, 2)),
              b.create<LLVM::BitcastOp>(
                  i16_ty, b.create<mlir::UnrealizedConversionCastOp>(
                               LLVM::getFixedVectorType(i8_ty, num_elements),
                               mlir::ValueRange{value})
                              .getResult(0)),
              zero_cst));
    } else {
      chunks = b.create<LLVM::BitcastOp>(
          chunks_ty, b.create<mlir::UnrealizedConversionCastOp>(
                          LLVM::getFixedVectorType(i8_ty, num_elements),
                          mlir::ValueRange{value})
                         .getResult(0));
    }

    llvm::SmallVector<mlir::Value, 4> results;
    mlir::StringAttr cvtIntr = b.getStringAttr(
        isFp8(value.getType().getElementType()) ? "llvm.amdgcn.cvt.pk.f32.fp8"
                                                : "llvm.amdgcn.cvt.pk.f32.bf8");
    mlir::Type result_ty = LLVM::getFixedVectorType(f32_ty, 2);
    LLVM::FastmathFlagsAttr flags =
        LLVM::FastmathFlagsAttr::get(b.getContext(), LLVM::FastmathFlags::ninf);
    for (size_t i = 0; i < num_elements / 2; i++) {
      mlir::Value chunk_pos = b.create<LLVM::ConstantOp>(i32_ty, (2 * i) / 4);
      mlir::Value chunk = b.create<LLVM::ExtractElementOp>(chunks, chunk_pos);
      LLVM::CallIntrinsicOp cvtOp = b.create<LLVM::CallIntrinsicOp>(
          result_ty, cvtIntr,
          mlir::ValueRange{
              chunk, b.create<LLVM::ConstantOp>(i1_ty, ((2 * i) % 4) != 0)},
          flags);

      results.push_back(
          b.create<LLVM::ExtractElementOp>(cvtOp.getResult(0), zero_cst));
      results.push_back(
          b.create<LLVM::ExtractElementOp>(cvtOp.getResult(0), one_cst));
    }

    if (to_ty.isF16()) {
      result_ty = LLVM::getFixedVectorType(b.getF16Type(), 2);
      cvtIntr = b.getStringAttr("llvm.amdgcn.cvt.pkrtz");
      for (size_t i = 0; i < num_elements / 2; i++) {
        LLVM::CallIntrinsicOp cvtOp = b.create<LLVM::CallIntrinsicOp>(
            result_ty, cvtIntr,
            mlir::ValueRange{results[2 * i], results[2 * i + 1]}, flags);

        results[2 * i] =
            b.create<LLVM::ExtractElementOp>(cvtOp.getResult(0), zero_cst);
        results[2 * i + 1] =
            b.create<LLVM::ExtractElementOp>(cvtOp.getResult(0), one_cst);
      }
    } else if (to_ty != f32_ty) {
      llvm::transform(results, results.begin(),
                      [&](mlir::Value v) -> mlir::Value {
                        return ConvertFromFloat(v, to_ty, b);
                      });
    }

    return results;
  }

  mlir::Value EmitExtFromF8Intrinsic(mlir::Value value, mlir::FloatType to_ty,
                                     mlir::ImplicitLocOpBuilder& b) const {
    assert(isFp8(value.getType()) || isBf8(value.getType()));

    mlir::FloatType f32_ty = b.getF32Type();
    mlir::IntegerType i32_ty = b.getI32Type();
    mlir::IntegerType i8_ty = b.getI8Type();
    mlir::Value zero_cst = b.create<LLVM::ConstantOp>(i32_ty, 0);
    // Emulate anyext
    mlir::Value input = b.create<LLVM::BitcastOp>(
        i32_ty, b.create<LLVM::InsertElementOp>(
                    b.create<LLVM::UndefOp>(LLVM::getFixedVectorType(i8_ty, 4)),
                    b.create<mlir::UnrealizedConversionCastOp>(
                         i8_ty, mlir::ValueRange{value})
                        .getResult(0),
                    zero_cst));
    mlir::StringAttr cvtIntr =
        b.getStringAttr(isFp8(value.getType()) ? "llvm.amdgcn.cvt.f32.fp8"
                                               : "llvm.amdgcn.cvt.f32.bf8");
    LLVM::FastmathFlagsAttr flags =
        LLVM::FastmathFlagsAttr::get(b.getContext(), LLVM::FastmathFlags::ninf);
    LLVM::CallIntrinsicOp cvtOp = b.create<LLVM::CallIntrinsicOp>(
        mlir::TypeRange{f32_ty}, cvtIntr, mlir::ValueRange{input, zero_cst},
        flags);

    return ConvertFromFloat(cvtOp.getResult(0), to_ty, b);
  }
};

class ConvertFloatAMDPass
    : public impl::ConvertFloatAMDPassBase<ConvertFloatAMDPass> {
 public:
  explicit ConvertFloatAMDPass(const ConvertFloatAMDPassOptions& options)
      : ConvertFloatAMDPassBase(options) {}

  explicit ConvertFloatAMDPass(const se::RocmComputeCapability& cc) : cc_(cc) {}

  void runOnOperation() override {
    if (!gpu_device_info_.empty()) {
      se::GpuDeviceInfoProto device_info;
      CHECK(tsl::protobuf::TextFormat::ParseFromString(gpu_device_info_,
                                                       &device_info));
      cc_ = se::DeviceDescription(device_info).rocm_compute_capability();
    }
    mlir::RewritePatternSet patterns(&getContext());
    bool nativeNanooFp8 = cc_.has_nanoo_fp8_support();
    patterns.add<RewriteFp8TruncFPattern, RewriteFp8ExtFPattern>(
        &getContext(), nativeNanooFp8);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  se::RocmComputeCapability cc_;
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConvertFloatAMDPass(
    const std::string& gpu_device_info) {
  ConvertFloatAMDPassOptions options;
  options.gpu_device_info_ = gpu_device_info;
  return std::make_unique<ConvertFloatAMDPass>(options);
}

std::unique_ptr<mlir::Pass> CreateConvertFloatAMDPass(
    const se::RocmComputeCapability& cc) {
  return std::make_unique<ConvertFloatAMDPass>(cc);
}

}  // namespace gpu
}  // namespace xla
