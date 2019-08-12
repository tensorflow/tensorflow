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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RUY_SPEC_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RUY_SPEC_H_

#include <limits>
#include <type_traits>

#include "tensorflow/lite/experimental/ruy/matrix.h"

namespace ruy {

// Our 'general' loop structure (the default) involves multi-threading and
// complicated loops aiming to optimize cache-friendliness. One may opt out of
// this and pick the 'simple' loop structure instead, which only performs well
// for small matrix sizes and only allows using one thread, in exchange for
// smaller code size.
enum class LoopStructure { kGeneral, kSimple, kAuto };

// In general we allow zero_point's to have any Scalar value. This is called
// 'asymmetric' quantization. We do take advantage of the optimization
// opportunities when zero_points happen at runtime to be 'symmetric' (e.g. the
// int8 value 0 or the uint8 value 128), but we still generate code to handle
// the general asymmetric case. By choosing kSymmetric here, one opts out of
// this and supports only the symmetric case, in exchange for smaller code size.
enum class ZeroPointSupport { kGeneral, kSymmetric };

// In general we allow all Layout's, even if we may use slow paths for some
// kinds of layouts. By choosing kRCC, one may opt out of this and
// only keep support for the simplest and most efficient combination of
// Layout's, in exchange for smaller code size. The case covered by
// kRCC is where the storage orders are exactly the following:
//    - LHS is RowMajor
//    - RHS is ColMajor
//    - Destination is ColMajor
enum class LayoutSupport { kGeneral, kRCC };

// A Spec describes all about a matrix multiplication operation that isn't
// encoded in the LHS, RHS and destination matrices. Some of that information
// is encoded as compile-time constants and types (for instance, the choice
// of accumulator type, AccumScalar). Some of that information is encoded as
// runtime values (for instance, the optional bias vector).
template <typename tAccumScalar, typename tDstScalar>
struct BasicSpec {
  // Accumulator type. The type of accumulators used to compute the dot-products
  // before being ultimately casted to the destination type.
  using AccumScalar = tAccumScalar;
  // The destination scalar type.
  using DstScalar = tDstScalar;
  // The bias vector data, if not null.
  const AccumScalar* bias = nullptr;
  // Only for non-floating-point cases. The fixed-point part (i.e. the mantissa)
  // of the multiplier by which accumulators are multiplied before being casted
  // to the destination type.
  AccumScalar multiplier_fixedpoint = 0;
  // Only for non-floating-point cases. The exponent part of the aforementioned
  // multiplier.
  int multiplier_exponent = 0;
  // Per-channel variant of multiplier_fixedpoint. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_fixedpoint.
  const AccumScalar* multiplier_fixedpoint_perchannel = nullptr;
  // Per-channel variant of multiplier_exponent. If not nullptr, this must
  // point to a buffer of as many values as there are rows in the destination
  // matrix. Each row of the destination matrix will use the corresponding
  // buffer element instead of multiplier_exponent.
  //
  // Either none or both of multiplier_exponent_perchannel and
  // multiplier_fixedpoint_perchannel must be nullptr.
  const int* multiplier_exponent_perchannel = nullptr;
  // min clamp bound of destination values.
  DstScalar clamp_min = std::is_floating_point<DstScalar>::value
                            ? -std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::lowest();
  // max clamp bound of destination values.
  DstScalar clamp_max = std::is_floating_point<DstScalar>::value
                            ? std::numeric_limits<DstScalar>::infinity()
                            : std::numeric_limits<DstScalar>::max();
  // See above enum LoopStructure
  static constexpr LoopStructure kLoopStructure = LoopStructure::kAuto;
  // See above enum LayoutSupport
  static constexpr LayoutSupport kLayoutSupport = LayoutSupport::kGeneral;
  // See above enum ZeroPointSupport
  static constexpr ZeroPointSupport kZeroPointSupport =
      ZeroPointSupport::kGeneral;
  // Testing-only, not meant to be used by actual users:
  // Used for testing of various kernel layouts.
  using StandardCppKernelLhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  using StandardCppKernelRhsLayout = FixedKernelLayout<Order::kColMajor, 1, 1>;
  // The value and even the meaning of this value are empirically
  // determined. Coarsely speaking, it's compared with the size of source
  // LHS and RHS operands to determine whether they are big enough to be worth
  // traversing in a more complicated "cache friendly" order. The current
  // value is roughly the minimum size of a L1 cache on any CPU that we
  // currently care about, e.g. ARM Cortex-A53. But we honestly don't even know
  // the precise extent to which this should be related to L1 cache size.
  //
  // A lower value is not necessarily 'safer' from a cache-friendliness
  // perspective: it means switching sooner (at smaller sizes) to more
  // complicated traversal orders, which might be adversarial to the CPU's
  // auto-prefetching or to the TLB.
  static int cache_friendly_traversal_threshold() { return 32 * 1024; }
};

}  // namespace ruy

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RUY_SPEC_H_
