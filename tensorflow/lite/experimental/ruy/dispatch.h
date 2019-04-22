/* Copyright 2019 Google LLC. All Rights Reserved.

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

// Translates a user-facing Mul call into an implementation-facing TrMul

// As a matrix multiplication library, Ruy offers a Mul entry point, performing
// matrix multiplication. For implementation purposes, it is much nicer to
// be dealing with the transpose-and-multiply operation, doing
//   Destination = Transpose(LHS) * RHS
// Indeed, the latter is performing dot-products between the *columns* of LHS
// and the columns of RHS, whereas a plain matrix multiplication is performing
// dot-products between the *rows* of LHS and the columns of RHS.
// That is why TrMul is nicer to implement, allowing for a more symmetric
// treatment of LHS and RHS.
//
// In this file, we translate a Mul call into a TrMul call by transposing the
// LHS, so that henceforth the deeper implementation layers only need to deal
// with TrMul.

// This file also selects between different TrMul versions specialized
// on the Path.

// This file also performs some checking of invariants to catch user errors.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_DISPATCH_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_DISPATCH_H_

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/common.h"
#include "tensorflow/lite/experimental/ruy/context.h"
#include "tensorflow/lite/experimental/ruy/impl.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/spec.h"

namespace ruy {

// If the Spec's LayoutSupport covers only some special cases,
// this function enforces that the matrix multiplication at hand falls into
// that special case.
template <typename Spec>
void EnforceLayoutSupport(const Layout& lhs_layout, const Layout& rhs_layout,
                          const Layout& dst_layout) {
  if (Spec::kLayoutSupport == LayoutSupport::kPackedLinearRCC) {
    RUY_DCHECK(IsPackedLinearRowMajor(lhs_layout));
    RUY_DCHECK(IsPackedLinearColMajor(rhs_layout));
    RUY_DCHECK(IsPackedLinearColMajor(dst_layout));
  }
}

template <typename Scalar>
bool IsSymmetricZeroPoint(Scalar zero_point) {
  return zero_point == SymmetricZeroPoint<Scalar>();
}

template <typename Spec, typename Scalar>
void CheckZeroPoint(Scalar zero_point) {
  if (std::is_floating_point<Scalar>::value ||
      Spec::kZeroPointSupport == ZeroPointSupport::kSymmetric) {
    RUY_DCHECK(IsSymmetricZeroPoint(zero_point));
  }
}

// If the Spec's ZeroPointSupport covers only some special cases,
// this function enforces that the matrix multiplication at hand falls into
// that special case.
template <typename Spec, typename LhsScalar, typename RhsScalar,
          typename DstScalar>
void EnforceZeroPointSupport(LhsScalar lhs_zero_point, RhsScalar rhs_zero_point,
                             DstScalar dst_zero_point) {
  CheckZeroPoint<Spec>(lhs_zero_point);
  CheckZeroPoint<Spec>(rhs_zero_point);
  CheckZeroPoint<Spec>(dst_zero_point);
}

// GetTrMulImplRunFn is implemented with template metaprogramming by mutual
// recursion between PathSearchCountdown and PathSearchCompiledPaths.
//
// GetTrMulImplRunFn is logically implementing the following computation:
//
// decltype(&TrMulImpl<...>::Run) GetTrMulImplRunFn(Path single_path) {
//   for (int bit = 8 * sizeof(Path) - 1; bit != -1; bit--) { // [1]
//     Path current_path = static_cast<Path>(1 << bit);
//     if ((CompiledPaths & current_path) != Path::kNone) { // [2]
//       if (current_path == single_path) { // [3]
//         return &TrMulImpl<current_path, ...>::Run;
//       }
//     }
//   }
//   return nullptr; // [4]
// }
//
//
//
// [1] - Done by the main definition of PathSearchCountdown. The `bit--` is
// done in the recursion of PathSearchOnlyCompiledPaths.
// [2] - Done by PathSearchOnlyCompiledPaths's partial template
// specialization on InCompiledPaths. This is the check which necessitates
// doing the whole computation at C++ compile time.
// [3] - Done by the `if` in the main definition of
// PathSearchOnlyCompiledPaths.
// [4] - Done by the partial specialization of PathSearchCountdown.
//
// The template metaprogramming is necessary because:
// - In `TrMulImpl<current_path, ...>::Run`, current_path must be a C++
// compile-time constant.
// - GetTrMulImplRunFn must not instantiate
// `TrMulImpl<curent_path, ...>::Run` for paths that are not in
// CompiledPaths, since that can result in bogus instantiations which cause
// a compile time failure.
template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename Spec>
struct PathSearchCountdown;

template <Path CompiledPaths, bool InCompiledPaths, int BitNumber,
          typename LhsScalar, typename RhsScalar, typename DstScalar,
          typename Spec>
struct PathSearchOnlyCompiledPaths {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static decltype(
      &TrMulImpl<Path::kNone, LhsScalar, RhsScalar, DstScalar, Spec>::Run)
  Search(Path single_path) {
    if (kCurrentPath == single_path) {
      return &TrMulImpl<kCurrentPath, LhsScalar, RhsScalar, DstScalar,
                        Spec>::Run;
    }
    return PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar,
                               RhsScalar, DstScalar, Spec>::Search(single_path);
  }
};

// Skip instantiating TrMulImpl if CompiledPaths doesn't contain the
// specified path.
template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename Spec>
struct PathSearchOnlyCompiledPaths<CompiledPaths, false, BitNumber, LhsScalar,
                                   RhsScalar, DstScalar, Spec> {
  static decltype(
      &TrMulImpl<Path::kNone, LhsScalar, RhsScalar, DstScalar, Spec>::Run)
  Search(Path single_path) {
    return PathSearchCountdown<CompiledPaths, BitNumber - 1, LhsScalar,
                               RhsScalar, DstScalar, Spec>::Search(single_path);
  }
};

template <Path CompiledPaths, int BitNumber, typename LhsScalar,
          typename RhsScalar, typename DstScalar, typename Spec>
struct PathSearchCountdown {
  static constexpr Path kCurrentPath = static_cast<Path>(1 << BitNumber);
  static decltype(
      &TrMulImpl<Path::kNone, LhsScalar, RhsScalar, DstScalar, Spec>::Run)
  Search(Path single_path) {
    return PathSearchOnlyCompiledPaths<
        CompiledPaths, (CompiledPaths & kCurrentPath) != Path::kNone, BitNumber,
        LhsScalar, RhsScalar, DstScalar, Spec>::Search(single_path);
  }
};

// Termination of the countdown. If the counter reaches -1, then we haven't
// found the specified path.
template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct PathSearchCountdown<CompiledPaths, -1, LhsScalar, RhsScalar, DstScalar,
                           Spec> {
  static decltype(
      &TrMulImpl<Path::kNone, LhsScalar, RhsScalar, DstScalar, Spec>::Run)
  Search(Path single_path) {
    return nullptr;
  }
};

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
decltype(&TrMulImpl<Path::kNone, LhsScalar, RhsScalar, DstScalar, Spec>::Run)
GetTrMulImplRunFn(Path single_path) {
  return PathSearchCountdown<CompiledPaths, 8 * sizeof(Path) - 1, LhsScalar,
                             RhsScalar, DstScalar, Spec>::Search(single_path);
};

template <Path CompiledPaths, typename LhsScalar, typename RhsScalar,
          typename DstScalar, typename Spec>
struct MulDispatch {
  void Mul(const Matrix<LhsScalar>& lhs, const Matrix<RhsScalar>& rhs,
           const Spec& spec, Context* context, Matrix<DstScalar>* dst) {
    gemmlowp::ScopedProfilingLabel label("Mul");

    const Path runtime_enabled_paths = context->GetRuntimeEnabledPaths();
    // The above query should resolve to specific paths, never return kNone.
    RUY_DCHECK(runtime_enabled_paths != Path::kNone);

    Path single_path =
        GetMostSignificantPath(CompiledPaths & runtime_enabled_paths);
    auto tr_mul_impl_run_fn =
        GetTrMulImplRunFn<CompiledPaths, LhsScalar, RhsScalar, DstScalar, Spec>(
            single_path);
    context->last_taken_path = single_path;

    EnforceLayoutSupport<Spec>(lhs.layout, rhs.layout, dst->layout);
    EnforceZeroPointSupport<Spec>(lhs.zero_point, rhs.zero_point,
                                  dst->zero_point);

    Matrix<LhsScalar> lhs_copy(lhs);
    Transpose(&lhs_copy);
    tr_mul_impl_run_fn(lhs_copy, rhs, spec, context, dst);
  }
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_DISPATCH_H_
