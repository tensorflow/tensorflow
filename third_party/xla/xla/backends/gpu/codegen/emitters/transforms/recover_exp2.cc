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

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "llvm/Support/Error.h"
/// ^^^ must be above others, as llvm don't strictly follow IWYU

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_RECOVEREXP2PASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

/// \brief Pattern rewriter to recover calls to math.exp2(x) from
/// math.exp(x * log(2)) expressions.
/// \details It's a bit of a rabbit's hole how to match data types properly due
/// to different rounding rules and the fact that at least now, data types lower
/// than fp32 are promoted to fp32 before coming into math.exp(). That includes
/// promotion of a value of log(2), so for bf16, a constant of type fp32 will
/// hold not a value of log(2) in fp32, but a value of log(2) in bf16 promoted
/// to fp32.
/// Current approach is the following: for different datatypes we prebuild a set
/// of log(2) representations and representation+/-1ulp (to accommodate for the
/// fact that precomputed constant in the code might have used a different
/// logarithm algorithm; that assumes +/-1 ulp is good enough to account that).
/// Then on a comparison stage, we identify a constant used as an exp() argument
/// multiplier, and compare it against all precomputed log(2) values. Types
/// wider than fp32 are narrowed to fp32 before the comparison (this makes a
/// compromise between a number of false positives and performance of the
/// matcher operation). This approach creates a narrow range of values that
/// triggers rewriting for types wider than fp32, and also for all types it
/// creates a small set of dedicated values, that trigger rewriting, which seems
/// a better compromise on false positive rate than taking a +/- 1ulp of the
/// smallest supported type value for all types. This can obviously be amended,
/// for example, types wider than fp16 could be narrowed to fp16 for comparison,
/// instead of float/fp32, if necessary.
/// \pre Canonicalization pass must happen before the rewriter
class RecoverExp2Pattern : public mlir::OpRewritePattern<mlir::math::ExpOp> {
  // note: make the widest type the first as it defines the canonical log2
  using SupportedNarrowTypes =
      std::tuple<mlir::Float32Type, mlir::FloatTF32Type, mlir::BFloat16Type
                 //, mlir::Float16Type // fp16 has the same 10bit mantissa as
                 // tf32 so no need to duplicate it. See commented out snippet
                 // in prefillLogs2From() for debugging aid when adding a new
                 // type
                 >;
  using WidestNarrowType = std::tuple_element_t<0, SupportedNarrowTypes>;

  template <typename TupleT, std::size_t Idx = 0>
      std::enable_if_t < Idx<std::tuple_size_v<TupleT>> prefillLogs2From(
                             const llvm::APFloat& main_log2) {
    using FloatT = std::tuple_element_t<Idx, TupleT>;

    auto f_obj = FloatT{};
    const auto& semantics = f_obj.getFloatSemantics();
    bool lossy{};
    llvm::APFloat log2_typed{main_log2};
    log2_typed.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &lossy);

    auto log2_next = log2_typed;
    log2_next.next(true);
    assert(log2_typed != log2_next);

    auto log2_prev = log2_typed;
    log2_prev.next(false);
    assert(log2_typed != log2_prev);

    static_assert(std::tuple_size_v<decltype(logs2_)> ==
                  3 * std::tuple_size_v<TupleT>);
    // preferring exact values to be nearer the beginning of the list
    logs2_[0 + Idx] = log2_typed.convertToFloat();
    logs2_[std::tuple_size_v<TupleT> + 2 * Idx + 0] =
        log2_next.convertToFloat();
    logs2_[std::tuple_size_v<TupleT> + 2 * Idx + 1] =
        log2_prev.convertToFloat();

    // the snippet below simply adding support for a new type by helping to
    // check what the values are and that they are unique. #include <cstdio>
    /*std::fprintf(stderr, "%d: %.8f, %.8f, %.8f\n", Idx, logs2_[0 + Idx],
                 logs2_[std::tuple_size_v<TupleT> + 2 * Idx + 0],
                 logs2_[std::tuple_size_v<TupleT> + 2 * Idx + 1]);*/

    if constexpr (Idx + 1 < std::tuple_size_v<TupleT>) {
      prefillLogs2From<TupleT, Idx + 1>(main_log2);
    }
  }

  template <typename Types>
  void initStaticsIfNeeded() {
    static_assert(std::tuple_size_v<Types> > 0);
    if (logs2_[0] == 0.) {
      auto log2 = llvm::APFloat(WidestNarrowType{}.getFloatSemantics());
      (void)log2.convertFromString("0.6931471805599453",
                                   llvm::APFloat::rmNearestTiesToEven);
      assert(log2.convertToFloat() == 0.6931471805599453F);
      prefillLogs2From<Types>(log2);
      assert(std::none_of(logs2_.begin(), logs2_.end(),
                          [](auto v) -> bool { return v == decltype(v){}; }));
    }
  }

 public:
  template <typename... PrmsT>
  explicit RecoverExp2Pattern(PrmsT&&... prms)
      : OpRewritePattern<mlir::math::ExpOp>(std::forward<PrmsT>(prms)...) {
    initStaticsIfNeeded<SupportedNarrowTypes>();
  }

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, mlir::PatternRewriter& rewriter) const override {
    // sanity check, perhaps not necessary at least now, but safer to leave
    if (!mlir::isa<mlir::FloatType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "must be a float type");
    }

    auto maybe_mul_op = op.getOperand().getDefiningOp<mlir::arith::MulFOp>();
    if (!maybe_mul_op) {
      return mlir::failure();
    }
    // preceding canonicalizer pass guarantees that if there's a constant
    // operand, it's the second
    auto maybe_const_op = maybe_mul_op.getOperand(1)
                              .getDefiningOp<mlir::arith::ConstantFloatOp>();
    if (!maybe_const_op) {
      return mlir::failure();
    }
    assert(mlir::isa<mlir::FloatType>(maybe_const_op.getType()));

    // Unfortunately we can't always call convertToFloat() as it doesn't work
    // for types wider than float, so to generalize better have to go through
    // the manual cropping of the widest type
    auto could_be_log_2 = maybe_const_op.value();
    bool lossy{};
    could_be_log_2.convert(WidestNarrowType{}.getFloatSemantics(),
                           llvm::APFloat::rmNearestTiesToEven, &lossy);

    // Also since there's no static or even non-const methods for RTTI support
    // in mlir/llvm now, doing a runtime check for actual operand type to only
    // check it against its log2 value and values of types that could be
    // promoted to it, is costlier than just checking against all values in
    // logs2_. This could slightly raise FP rate, but hopefully this shouldn't
    // be a problem.
    if (std::any_of(logs2_.begin(), logs2_.end(),
                    [could_be = could_be_log_2.convertToFloat()](
                        auto v) -> bool { return v == could_be; })) {
      rewriter.replaceOpWithNewOp<mlir::math::Exp2Op>(
          op, maybe_mul_op.getOperand(0));
      // no need to do anything with deletion of other ops, llvm will delete
      // those that have no uses automatically.
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "The constant didn't match expected log(2) representation");
  }

 private:
  /// storage for precomputed log(2) cropped to different types precision +/-
  /// 1ulp. We need something directly iterable by index to account for possible
  /// type promotions in the code.
  inline static std::array<float, 3 * std::tuple_size_v<SupportedNarrowTypes>>
      logs2_{};
};

class RecoverExp2Pass : public impl::RecoverExp2PassBase<RecoverExp2Pass> {
 public:
  using RecoverExp2PassBase::RecoverExp2PassBase;
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RecoverExp2Pattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

/// \brief Create a pass to recover calls to math.exp2(x) from
/// math.exp(x * log(2)) expressions.
/// \pre Canonicalization pass must happen before this pass
std::unique_ptr<mlir::Pass> CreateRecoverExp2Pass() {
  return std::make_unique<RecoverExp2Pass>();
}

}  // namespace gpu
}  // namespace xla
