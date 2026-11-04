/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/literal_util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/numeric/int128.h"
#include "absl/random/uniform_int_distribution.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/index_util.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/constraint_state.h"
#include "xla/tsl/lib/core/bitmap.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {
namespace {

using absl::StrCat;

// Return a literal with all arrays of type FromNativeT converted to type
// ToNativeT in the given literal.
template <typename FromNativeT, typename ToNativeT>
Literal ConvertType(LiteralSlice literal) {
  // First construct shape of the result.
  Shape result_shape(literal.shape());
  ShapeUtil::ForEachMutableSubshape(
      &result_shape, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->element_type() ==
            primitive_util::NativeToPrimitiveType<FromNativeT>()) {
          subshape->set_element_type(
              primitive_util::NativeToPrimitiveType<ToNativeT>());
        }
      });
  Literal result(result_shape);

  // Then copy over the data from 'literal' converting FromNativeT values to
  // ToNativeT values as necessary.
  ShapeUtil::ForEachSubshape(
      literal.shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (subshape.IsArray()) {
          if (subshape.element_type() ==
              primitive_util::NativeToPrimitiveType<FromNativeT>()) {
            auto src = literal.data<FromNativeT>(shape_index);
            auto dest = result.data<ToNativeT>(shape_index);
            for (int64_t i = 0, end = src.size(); i < end; ++i) {
              dest[i] = static_cast<ToNativeT>(src[i]);
            }
          } else {
            CHECK_OK(result.CopyFrom(literal,
                                     /*dest_shape_index=*/shape_index,
                                     /*src_shape_index=*/shape_index));
          }
        }
      });
  return result;
}

template <PrimitiveType kType>
using NativeT = typename primitive_util::PrimitiveTypeToNative<kType>::type;

template <PrimitiveType kType, typename F, typename... Args>
Literal CreateScalarImpl(F&& value_provider, Args... args) {
  return LiteralUtil::CreateR0<NativeT<kType>>(
      value_provider(std::forward<Args>(args)...));
}

template <template <PrimitiveType> class F, typename... Args>
Literal CreateScalar(PrimitiveType primitive_type, Args... args) {
  return primitive_util::PrimitiveTypeSwitch<Literal>(
      [&](auto primitive_type_constant) -> Literal {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          return CreateScalarImpl<primitive_type_constant>(
              F<primitive_type_constant>{}, std::forward<Args>(args)...);
        }
        LOG(FATAL) << "Unhandled primitive type " << primitive_type;
      },
      primitive_type);
}

template <PrimitiveType kType>
struct ZeroProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(0); }
};

// Use template specialization for the E8M0 type, as it has no zero
// representation, so static_cast<> returns NaN. The actual zero-like value
// is 2^-127.
template <>
struct ZeroProvider<F8E8M0FNU> {
  NativeT<F8E8M0FNU> operator()() const {
    return Eigen::numext::bit_cast<NativeT<F8E8M0FNU>>('\0');
  }
};

template <PrimitiveType kType>
struct OneProvider {
  NativeT<kType> operator()() const { return static_cast<NativeT<kType>>(1); }
};

template <typename T>
struct IsReal {
  static constexpr bool value = std::numeric_limits<T>::is_specialized;
};

template <typename T>
struct IsValidScalarType {
  static constexpr bool value = IsReal<T>::value || is_complex_v<T>;
};

template <typename NativeT>
NativeT GetMaxImpl() {
  if constexpr (IsReal<NativeT>::value) {
    if constexpr (std::numeric_limits<NativeT>::has_infinity) {
      return std::numeric_limits<NativeT>::infinity();
    }
    return std::numeric_limits<NativeT>::max();
  }
  LOG(FATAL) << "No max value for given type.";
}

template <typename NativeT>
NativeT GetMaxFiniteImpl() {
  if constexpr (IsReal<NativeT>::value) {
    return std::numeric_limits<NativeT>::max();
  }
  LOG(FATAL) << "No finite max value for given type.";
}

template <typename NativeT>
NativeT GetMinImpl() {
  if constexpr (IsReal<NativeT>::value) {
    if constexpr (std::numeric_limits<NativeT>::has_infinity) {
      return -std::numeric_limits<NativeT>::infinity();
    }
    return std::numeric_limits<NativeT>::lowest();
  }
  LOG(FATAL) << "No min value for given type.";
}

template <PrimitiveType kType>
struct MaxProvider {
  NativeT<kType> operator()() const { return GetMaxImpl<NativeT<kType>>(); }
};

template <PrimitiveType kType>
struct MaxFiniteProvider {
  NativeT<kType> operator()() const {
    return GetMaxFiniteImpl<NativeT<kType>>();
  }
};

template <PrimitiveType kType>
struct MinProvider {
  NativeT<kType> operator()() const { return GetMinImpl<NativeT<kType>>(); }
};

template <typename NativeT>
std::enable_if_t<IsReal<NativeT>::value, NativeT> GetMaxElementImpl(
    const LiteralBase& literal) {
  auto view = literal.data<NativeT>();
  return *absl::c_max_element(view);
}

template <typename NativeT>
std::enable_if_t<!IsReal<NativeT>::value, NativeT> GetMaxElementImpl(
    const LiteralBase& literal) {
  LOG(FATAL) << "Unsupported type.";
}

template <PrimitiveType kType>
struct MaxElementProvider {
  NativeT<kType> operator()(const LiteralBase& literal) const {
    return GetMaxElementImpl<NativeT<kType>>(literal);
  }
};

template <typename FloatT>
void PopulateWithIntNext(Literal* literal) {
  using BitRepT = UnsignedIntegerTypeForSizeType<sizeof(FloatT)>;
  // Duplicates may be generated if we don't have enough bits.
  // Skip bfloat16 and float32 subnormals.
  const FloatT kFirstValue =
      std::is_same_v<FloatT, bfloat16> || sizeof(FloatT) >= sizeof(float)
          ? std::numeric_limits<FloatT>::min()
          : std::numeric_limits<FloatT>::denorm_min();
  // `current` keeps track of the next value we need to populate.
  auto current = literal->data<FloatT>().begin();
  auto end = literal->data<FloatT>().end();
  // `sign` keeps track of the sign of the next value.
  bool sign = false;
  while (current != end) {
    // We start populating values at zero and increase magnitude from there.
    *current = sign ? static_cast<FloatT>(-0.0f) : static_cast<FloatT>(0.0f);
    current++;
    // The next value is either the smallest denormal or normal.
    auto value = sign ? -kFirstValue : kFirstValue;
    // Fill the array with values of increasing magnitude until we hit a
    // non-finite value.
    while (current != end && Eigen::numext::isfinite(value)) {
      // Populate the value.
      *current = value;
      // Generate the next value by lexicographically increasing the bit
      // representation.
      const BitRepT next_value = Eigen::numext::bit_cast<BitRepT>(value) + 1;
      value = Eigen::numext::bit_cast<FloatT>(next_value);
      current++;
    }
    // We ran out of finite values, flip the sign and begin again.
    sign = !sign;
  }
}

template <typename FloatT>
void PopulateWithNoDuplicateData(Literal* literal, std::minstd_rand0* engine) {
  PopulateWithIntNext<FloatT>(literal);
  std::shuffle(literal->data<FloatT>().begin(), literal->data<FloatT>().end(),
               *engine);
}

// Number of distinct outputs produced by a single std::minstd_rand0 transition
// in [std::minstd_rand0::min(), std::minstd_rand0::max()] = [1, 2^31 - 2].
constexpr uint64_t kMinstdSpan =
    static_cast<uint64_t>(std::minstd_rand0::max() - std::minstd_rand0::min()) +
    1;
constexpr uint64_t kMinstdSpanSquared = kMinstdSpan * kMinstdSpan;

// Draws one zero-based sample in [0, kMinstdSpan) from `engine`, advancing
// `engine` by exactly one transition.
uint64_t DrawMinstdZeroBased(std::minstd_rand0* engine) {
  return static_cast<uint64_t>((*engine)() - std::minstd_rand0::min());
}

// Stateless SplitMix64 finalizer used to deterministically derive fallback
// candidate values (when a sampled value hits a forbidden state such as 0.0,
// NaN, or Inf) without advancing the per-partition engine past its fixed
// `steps_per_element` transition budget.
uint64_t Mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

// Maps one zero-based std::minstd_rand0 draw in [0, kMinstdSpan) uniformly to a
// double in [0.0, 1.0). Sufficient for 32-bit and narrower floating-point
// significands (<= 24 bits).
double UniformDouble01From1Draw(uint64_t draw0) {
  return static_cast<double>(draw0) * (1.0 / static_cast<double>(kMinstdSpan));
}

// Combines two zero-based std::minstd_rand0 draws into a 53-bit IEEE 754 double
// significand in [0.0, 1.0) so 64-bit doubles receive full mantissa entropy.
double UniformDouble01From2Draws(uint64_t draw0, uint64_t draw1) {
  const uint64_t combined = draw0 * kMinstdSpan + draw1;
  const uint64_t bits53 = static_cast<uint64_t>(
      (static_cast<absl::uint128>(combined) << 53) / kMinstdSpanSquared);
  return static_cast<double>(bits53) *
         (1.0 / static_cast<double>(uint64_t{1} << 53));
}

// Combines two zero-based std::minstd_rand0 draws into an unbiased 32-bit
// unsigned integer in [0, 2^32).
uint32_t UniformUint32From2Draws(uint64_t draw0, uint64_t draw1) {
  const uint64_t combined = draw0 * kMinstdSpan + draw1;
  return static_cast<uint32_t>((static_cast<absl::uint128>(combined) << 32) /
                               kMinstdSpanSquared);
}

// Advances a std::minstd_rand0 engine by `steps` transitions in O(log(steps))
// time using binary modular exponentiation. Since std::minstd_rand0 is a
// multiplicative linear congruential generator with transition
// s_{k+1} = (s_k * 16807) mod (2^31 - 1), jumping ahead by `steps` allows
// worker threads to populate disjoint slices of a large tensor concurrently
// while advancing the caller's engine deterministically.
std::minstd_rand0 AdvanceMinstdRand0(std::minstd_rand0 engine, uint64_t steps) {
  if (steps == 0) {
    return engine;
  }
  constexpr uint64_t kModulus = std::minstd_rand0::modulus;
  constexpr uint64_t kMultiplier = std::minstd_rand0::multiplier;
  // Advance once via operator() to normalize seed 0 (or any unnormalized state)
  // into [1, kModulus - 1], then multiply by kMultiplier^(steps - 1) mod
  // kModulus.
  uint64_t state = engine();
  uint64_t remaining = steps - 1;
  uint64_t base = kMultiplier;
  uint64_t factor = 1;
  while (remaining > 0) {
    if (remaining & 1) {
      factor = (factor * base) % kModulus;
    }
    base = (base * base) % kModulus;
    remaining >>= 1;
  }
  state = (state * factor) % kModulus;
  return std::minstd_rand0(static_cast<std::minstd_rand0::result_type>(state));
}

// Minimum number of elements per parallel partition. Tensors smaller than
// kMinElementsPerPartition use the legacy single-threaded distribution loops to
// preserve bit-identical random sequences in small unit tests. Tensors in
// [kMinElementsPerPartition, 2 * kMinElementsPerPartition) run the fixed-step
// generator inline on the calling thread, and tensors of at least
// 2 * kMinElementsPerPartition partition work across the global
// ForEachIndexParallel thread pool.
constexpr int64_t kMinElementsPerPartition = 16384;

// Partitions `num_elements` across the global ForEachIndexParallel thread pool,
// giving each partition its own deterministically jumped-ahead engine state.
// Every element in `fn` must advance `local_engine` by strictly
// `steps_per_element` transitions so output values and the final `*engine`
// state are independent of the host thread count and partition boundaries.
// Tensors below 2 * kMinElementsPerPartition run inline to avoid thread
// dispatch overhead.
template <typename PerPartitionFn>
void ParallelPopulateWithEngine(int64_t num_elements, std::minstd_rand0* engine,
                                int64_t steps_per_element,
                                PerPartitionFn&& fn) {
  if (num_elements <= 0) {
    return;
  }
  const int max_threads = ShapeUtil::GetForEachIndexParallelThreadCount();
  if (num_elements < kMinElementsPerPartition * 2 || max_threads <= 1) {
    fn(0, num_elements, engine);
    return;
  }
  const int64_t num_partitions = std::min<int64_t>(
      max_threads,
      CeilOfRatio<int64_t>(num_elements, kMinElementsPerPartition));
  const int64_t partition_size =
      CeilOfRatio<int64_t>(num_elements, num_partitions);
  const std::minstd_rand0 base_engine = *engine;
  const uint64_t steps_u64 = static_cast<uint64_t>(steps_per_element);
  Shape work_shape = ShapeUtil::MakeShape(U8, {num_partitions});
  ShapeUtil::ForEachIndexParallel(
      work_shape,
      [&](absl::Span<const int64_t> partition_index,
          int /*thread_id*/) -> absl::StatusOr<bool> {
        const int64_t partition_id = partition_index[0];
        const int64_t start = partition_id * partition_size;
        const int64_t end = std::min(start + partition_size, num_elements);
        if (start < end) {
          std::minstd_rand0 local_engine = AdvanceMinstdRand0(
              base_engine, static_cast<uint64_t>(start) * steps_u64);
          fn(start, end, &local_engine);
        }
        return true;
      });
  *engine = AdvanceMinstdRand0(base_engine,
                               static_cast<uint64_t>(num_elements) * steps_u64);
}

// Populates a floating point literal with random floating points sampled from a
// uniform-log distribution spanning approximately the entire range of the
// representable floating point.
template <typename FloatT>
void PopulateWithRandomFullRangeFloatingPointData(Literal* literal,
                                                  std::minstd_rand0* engine) {
  constexpr float kSpecialValueProbability = 1e-6;
  constexpr float kSpecialValues[] = {+0.F,
                                      -0.F,
                                      1.F,
                                      -1.F,
                                      std::numeric_limits<float>::infinity(),
                                      -std::numeric_limits<float>::infinity()};
  constexpr int kNumSpecialValues = sizeof(kSpecialValues) / sizeof(float);
  constexpr float kHalfInterval =
      0.5f + (kSpecialValueProbability * kNumSpecialValues) / 2;

  // Generates floating points with a log-uniform distribution. This causes the
  // exponent of the floating point to have a uniform distribution.
  const int min_exp = std::numeric_limits<FloatT>::min_exponent;
  const int max_exp = std::numeric_limits<FloatT>::max_exponent;

  using GeneratorT =
      std::conditional_t<sizeof(FloatT) >= sizeof(double), double, float>;
  absl::Span<FloatT> data = literal->data<FloatT>();
  if (static_cast<int64_t>(data.size()) < kMinElementsPerPartition) {
    std::uniform_real_distribution<float> special_value_gen(0, 1);
    std::uniform_real_distribution<GeneratorT> generator(min_exp - 1,
                                                         max_exp - 1);
    for (FloatT& value : data) {
      float random_interval = special_value_gen(*engine);
      if (random_interval < kSpecialValueProbability * kNumSpecialValues) {
        value = static_cast<FloatT>(
            kSpecialValues[(*engine)() % kNumSpecialValues]);
      } else {
        const float sign = (random_interval < kHalfInterval) ? 1 : -1;
        value = static_cast<FloatT>(std::exp2(generator(*engine)) * sign);
      }
    }
    return;
  }

  // Always consume a fixed number of RNG transitions per element (3 for 64-bit
  // floats, 2 for <= 32-bit floats) regardless of whether the element hits the
  // special-value branch or the normal log-uniform branch so parallel
  // partitions stay synchronized.
  constexpr bool kNeedsTwoDrawsForExp = sizeof(GeneratorT) >= sizeof(double);
  constexpr int64_t kStepsPerElement = kNeedsTwoDrawsForExp ? 3 : 2;

  const double min_exp_d = static_cast<double>(min_exp - 1);
  const double max_exp_d = static_cast<double>(max_exp - 1);
  ParallelPopulateWithEngine(
      static_cast<int64_t>(data.size()), engine, kStepsPerElement,
      [&](int64_t start, int64_t end, std::minstd_rand0* local_engine) {
        for (int64_t i = start; i < end; ++i) {
          const uint64_t draw0 = DrawMinstdZeroBased(local_engine);
          const uint64_t draw1 = DrawMinstdZeroBased(local_engine);
          uint64_t draw2 = 0;
          if constexpr (kNeedsTwoDrawsForExp) {
            draw2 = DrawMinstdZeroBased(local_engine);
          }
          const float random_interval =
              static_cast<float>(UniformDouble01From1Draw(draw0));
          if (random_interval < kSpecialValueProbability * kNumSpecialValues) {
            data[i] =
                static_cast<FloatT>(kSpecialValues[draw1 % kNumSpecialValues]);
          } else {
            const float sign = (random_interval < kHalfInterval) ? 1.0f : -1.0f;
            const double u_exp = kNeedsTwoDrawsForExp
                                     ? UniformDouble01From2Draws(draw1, draw2)
                                     : UniformDouble01From1Draw(draw1);
            const GeneratorT exp_val = static_cast<GeneratorT>(
                (1.0 - u_exp) * min_exp_d + u_exp * max_exp_d);
            data[i] = static_cast<FloatT>(std::exp2(exp_val) * sign);
          }
        }
      });
}

template <typename FloatT, typename GeneratorT>
void PopulateWithRandomFloatingPointData(
    Literal* literal, std::minstd_rand0* engine,
    std::optional<ConstraintInterval> interval) {
  GeneratorT min = std::max<GeneratorT>(
      static_cast<GeneratorT>(-0.1),
      static_cast<GeneratorT>(std::numeric_limits<FloatT>::lowest()));
  GeneratorT max = std::min<GeneratorT>(
      static_cast<GeneratorT>(0.2),
      static_cast<GeneratorT>(std::numeric_limits<FloatT>::max()));
  if (interval.has_value() && !interval->IsUnconstrained()) {
    min = interval->min == ConstraintInterval::kMin
              ? min
              : static_cast<GeneratorT>(interval->min);
    max = interval->max == ConstraintInterval::kMax
              ? max
              : static_cast<GeneratorT>(interval->max);

    // Prevent Undefined Behavior by fixing inverted ranges.
    // If a single explicit bound crossed the opposite default bound,
    // shift the unconstrained bound to maintain a 0.3 generation spread.
    if (interval->max == ConstraintInterval::kMax && min > max) {
      max = min + static_cast<GeneratorT>(0.3);
    } else if (interval->min == ConstraintInterval::kMin && max < min) {
      min = max - static_cast<GeneratorT>(0.3);
    }
  }
  absl::Span<FloatT> data = literal->data<FloatT>();
  if (min == max) {
    std::fill(data.begin(), data.end(), static_cast<FloatT>(min));
    return;
  }
  const bool exclude_zero = interval.has_value() && interval->exclude_zero;
  if (static_cast<int64_t>(data.size()) < kMinElementsPerPartition) {
    std::uniform_real_distribution<GeneratorT> generator(min, max);
    for (FloatT& value : data) {
      value = static_cast<FloatT>(generator(*engine));
      while (exclude_zero && value == static_cast<FloatT>(0.0f)) {
        value = static_cast<FloatT>(generator(*engine));
      }
    }
    return;
  }
  const double min_d = static_cast<double>(min);
  const double max_d = static_cast<double>(max);
  constexpr bool kNeedsTwoDraws = sizeof(FloatT) >= sizeof(double);
  constexpr int64_t kStepsPerElement = kNeedsTwoDraws ? 2 : 1;
  ParallelPopulateWithEngine(
      static_cast<int64_t>(data.size()), engine, kStepsPerElement,
      [&](int64_t start, int64_t end, std::minstd_rand0* local_engine) {
        for (int64_t i = start; i < end; ++i) {
          const uint64_t draw0 = DrawMinstdZeroBased(local_engine);
          uint64_t draw1 = 0;
          if constexpr (kNeedsTwoDraws) {
            draw1 = DrawMinstdZeroBased(local_engine);
          }
          double u = kNeedsTwoDraws ? UniformDouble01From2Draws(draw0, draw1)
                                    : UniformDouble01From1Draw(draw0);
          // Convex interpolation avoids (max - min) overflow to +Inf on wide
          // double intervals such as [-1e308, 1e308].
          GeneratorT scaled =
              static_cast<GeneratorT>((1.0 - u) * min_d + u * max_d);
          scaled = std::min(max, std::max(min, scaled));
          FloatT value = static_cast<FloatT>(scaled);
          if (exclude_zero && value == static_cast<FloatT>(0.0f)) {
            // Derive retry candidates via stateless Mix64 so the per-element
            // engine step count remains constant across parallel partitions.
            uint64_t retry_state = draw0 ^ (draw1 << 31);
            for (int attempt = 0;
                 attempt < 64 && value == static_cast<FloatT>(0.0f);
                 ++attempt) {
              retry_state = Mix64(retry_state + 0x9e3779b97f4a7c15ULL);
              u = static_cast<double>(retry_state >> 11) *
                  (1.0 / static_cast<double>(uint64_t{1} << 53));
              scaled = static_cast<GeneratorT>((1.0 - u) * min_d + u * max_d);
              scaled = std::min(max, std::max(min, scaled));
              value = static_cast<FloatT>(scaled);
            }
            if (value == static_cast<FloatT>(0.0f)) {
              const FloatT max_f = static_cast<FloatT>(max);
              value = (max_f != static_cast<FloatT>(0.0f))
                          ? max_f
                          : static_cast<FloatT>(min);
            }
          }
          data[i] = value;
        }
      });
}

// Reinterprets the low-order bits of `bits` as a floating-point value of type
// `FloatT` (1, 2, or 4 bytes).
template <typename FloatT>
FloatT BitCastUint32ToFloat(uint32_t bits) {
  if constexpr (sizeof(FloatT) == 1) {
    return absl::bit_cast<FloatT>(static_cast<uint8_t>(bits));
  } else if constexpr (sizeof(FloatT) == 2) {
    return absl::bit_cast<FloatT>(static_cast<uint16_t>(bits));
  } else if constexpr (sizeof(FloatT) == 4) {
    return absl::bit_cast<FloatT>(bits);
  } else {
    LOG(FATAL) << "Unsupported float size: " << sizeof(FloatT);
  }
}

// Populates the literal with uniformly distributed full-range floating point
// data. The difference from PopulateWithRandomFullRangeFloatingPointData is
// that it samples uniformly from all possible values of the float (including
// subnormals, 0, etc). It excludes inf and nan.
template <typename FloatT>
void PopulateWithUniformFullRangeFloatingPointData(Literal* literal,
                                                   std::minstd_rand0* engine) {
  CHECK_LE(sizeof(FloatT), 4) << "Only floats up to 4 bytes supported.";
  // Needed for C++ template.
  const int64_t bit_width =
      std::min(32, primitive_util::BitWidth(
                       primitive_util::NativeToPrimitiveType<FloatT>()));
  const uint32_t max_val = bit_width == 32
                               ? std::numeric_limits<uint32_t>::max()
                               : (1U << bit_width) - 1;
  absl::Span<FloatT> data = literal->data<FloatT>();
  if (static_cast<int64_t>(data.size()) < kMinElementsPerPartition) {
    absl::uniform_int_distribution<uint32_t> generator(0, max_val);
    for (FloatT& value : data) {
      int repeats = 0;
      while (true) {
        value = BitCastUint32ToFloat<FloatT>(generator(*engine));
        if (!std::isinf(value) && !std::isnan(value)) {
          break;
        }
        repeats++;
        CHECK_LE(repeats, 5)
            << "Exceeded max_repeats=5 when generating float values";
      }
    }
    return;
  }
  const int64_t steps_per_element = bit_width > 16 ? 2 : 1;
  ParallelPopulateWithEngine(
      static_cast<int64_t>(data.size()), engine, steps_per_element,
      [&](int64_t start, int64_t end, std::minstd_rand0* local_engine) {
        for (int64_t i = start; i < end; ++i) {
          const uint64_t draw0 = DrawMinstdZeroBased(local_engine);
          uint64_t draw1 = 0;
          uint32_t raw_bits = 0;
          if (steps_per_element == 2) {
            draw1 = DrawMinstdZeroBased(local_engine);
            raw_bits = UniformUint32From2Draws(draw0, draw1) & max_val;
          } else {
            raw_bits = static_cast<uint32_t>(draw0) & max_val;
          }
          FloatT value = BitCastUint32ToFloat<FloatT>(raw_bits);
          if (std::isinf(value) || std::isnan(value)) {
            // Re-sample using stateless Mix64 so the engine advances by a fixed
            // number of transitions per element regardless of NaN/Inf retries.
            uint64_t retry_state = draw0 ^ (draw1 << 31);
            for (int attempt = 0;
                 attempt < 64 && (std::isinf(value) || std::isnan(value));
                 ++attempt) {
              retry_state = Mix64(retry_state + 0x9e3779b97f4a7c15ULL);
              raw_bits = static_cast<uint32_t>(retry_state) & max_val;
              value = BitCastUint32ToFloat<FloatT>(raw_bits);
            }
            if (std::isinf(value) || std::isnan(value)) {
              value = static_cast<FloatT>(0.0f);
            }
          }
          data[i] = value;
        }
      });
}

template <typename FloatT>
void PopulateWithFloatingPointData(
    Literal* literal, std::minstd_rand0* engine, bool no_duplicates,
    bool use_large_range, std::optional<int64_t> max_bits_of_precision,
    std::function<double(std::minstd_rand0*)> generator = nullptr,
    std::optional<ConstraintInterval> interval = std::nullopt) {
  using ComputeT =
      std::conditional_t<sizeof(FloatT) < sizeof(float), float, FloatT>;
  CHECK_NOTNULL(engine);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<FloatT>());
  const int64_t bit_width =
      primitive_util::BitWidth(primitive_util::NativeToPrimitiveType<FloatT>());
  // Floats with exceptionally small bit width can represent just a handful of
  // values. No sense in restricting the output values in the way we do for
  // higher precision formats. We just let the random generator do its thing.
  const bool float_with_small_bit_width = (bit_width <= 4);
  if (generator != nullptr) {
    for (FloatT& value : literal->data<FloatT>()) {
      value = static_cast<FloatT>(generator(engine));
    }
  } else if (max_bits_of_precision.has_value()) {
    CHECK(!use_large_range) << "Cannot set both use_large_range and "
                               "max_bits_of_precision for floating points.";
    CHECK(!no_duplicates) << "Cannot set both no_duplicates and "
                             "max_bits_of_precision for floating points.";
    absl::uniform_int_distribution<int64_t> generator(
        -(1 << *max_bits_of_precision), 1 << *max_bits_of_precision);
    for (FloatT& value : literal->data<FloatT>()) {
      int64_t temp = generator(*engine);
      // We want to generate floating point numbers to a fixed precision, while
      // keeping them between -1 and 1. This preserves their bits of precision
      // while keeping the numbers small.
      value = static_cast<FloatT>(
          temp == 0 ? 0 : temp * pow(2, -ceil(log2(abs(temp)))));
    }
  } else if (no_duplicates) {
    PopulateWithNoDuplicateData<FloatT>(literal, engine);
  } else if (float_with_small_bit_width) {
    PopulateWithUniformFullRangeFloatingPointData<FloatT>(literal, engine);
  } else if (use_large_range) {
    PopulateWithRandomFullRangeFloatingPointData<FloatT>(literal, engine);
  } else {
    PopulateWithRandomFloatingPointData<FloatT, ComputeT>(literal, engine,
                                                          interval);
  }
}

template <typename ComplexT>
void PopulateWithComplexData(
    Literal* result, std::minstd_rand0* engine, bool no_duplicates,
    bool use_large_range,
    std::function<double(std::minstd_rand0*)> generator = nullptr) {
  using InnerFloatT = typename ComplexT::value_type;
  CHECK_NOTNULL(engine);
  CHECK_EQ(result->shape().element_type(),
           primitive_util::NativeToPrimitiveType<ComplexT>());
  Shape floating_point_shape = ShapeUtil::ChangeElementType(
      result->shape(), primitive_util::NativeToPrimitiveType<InnerFloatT>());
  Literal real_lit(floating_point_shape);
  Literal imaginary_lit(floating_point_shape);

  PopulateWithFloatingPointData<InnerFloatT>(
      &real_lit, engine, no_duplicates, use_large_range,
      /*max_bits_of_precision=*/std::nullopt, generator);
  PopulateWithFloatingPointData<InnerFloatT>(
      &imaginary_lit, engine, no_duplicates, use_large_range,
      /*max_bits_of_precision=*/std::nullopt, generator);

  absl::Span<const InnerFloatT> real_data = real_lit.data<InnerFloatT>();
  absl::Span<const InnerFloatT> imaginary_data =
      imaginary_lit.data<InnerFloatT>();
  absl::Span<ComplexT> result_data = result->data<ComplexT>();
  for (int i = 0; i < real_lit.data<InnerFloatT>().size(); i++) {
    result_data[i] = ComplexT(real_data[i], imaginary_data[i]);
  }
}

template <typename TargetT>
TargetT SafeClampInt64(int64_t value) {
  static_assert(std::numeric_limits<TargetT>::is_integer,
                "TargetT must be an integral type.");
  if constexpr (!std::numeric_limits<TargetT>::is_signed) {
    if (value <= 0) {
      return TargetT{0};
    }
    uint64_t uval = static_cast<uint64_t>(value);
    if (uval > static_cast<uint64_t>(std::numeric_limits<TargetT>::max())) {
      return std::numeric_limits<TargetT>::max();
    }
    return static_cast<TargetT>(uval);
  } else {
    if (value < static_cast<int64_t>(std::numeric_limits<TargetT>::lowest())) {
      return std::numeric_limits<TargetT>::lowest();
    }
    if (value > static_cast<int64_t>(std::numeric_limits<TargetT>::max())) {
      return std::numeric_limits<TargetT>::max();
    }
    return static_cast<TargetT>(value);
  }
}

// uniform_int_distribution is not defined for 8-bit integers.
// Use 'short' for those types.
template <typename IntT>
using RngT = std::conditional_t<
    sizeof(IntT) < sizeof(uint16_t),
    std::conditional_t<std::numeric_limits<IntT>::is_signed, int16_t, uint16_t>,
    IntT>;

// Computes safe [min, max] bounds for integral literal generation.
// - If no limit is specified, returns the full type range [lowest, max].
// - If use_large_range is true or bit_width <= 4, clamps the limit directly.
// - Otherwise, bounds values to B/2 bits to prevent hardware ALU overflow
//   (multiplication, squaring, etc.) and float conversion explosions.
// - Expands to full range if no_duplicates requires more unique elements than
//   B/2 capacity.
template <typename IntT>
std::pair<IntT, IntT> GetIntegralBounds(
    const Shape& shape, bool use_large_range, bool no_duplicates,
    std::optional<std::pair<int64_t, int64_t>> limit) {
  if (!limit.has_value()) {
    return {std::numeric_limits<IntT>::lowest(),
            std::numeric_limits<IntT>::max()};
  }

  constexpr int64_t bit_width = sizeof(IntT) * 8;

  // Sub-byte integers (<= 4 bits) already have tiny domains (<= 16 values).
  if (use_large_range || bit_width <= 4) {
    return {SafeClampInt64<IntT>(limit->first),
            SafeClampInt64<IntT>(limit->second)};
  }

  // Calculate default B/2 bitwidth bounds and default_range_size (2^H - 1),
  // which is the width of the domain [default_min, default_max].
  int64_t h = bit_width / 2;
  int64_t default_min;
  int64_t default_max;
  int64_t default_range_size;
  if constexpr (std::numeric_limits<IntT>::is_signed) {
    default_min = -(int64_t{1} << (h - 1));
    default_max = (int64_t{1} << (h - 1)) - 1;
    default_range_size = (int64_t{1} << h) - 1;
  } else {
    default_min = 0;
    default_max = (int64_t{1} << h) - 1;
    default_range_size = default_max;
  }

  // If no_duplicates is requested, ensure capacity >= element count.
  int64_t num_elements = ShapeUtil::ElementsIn(shape);
  if (no_duplicates && num_elements > default_range_size) {
    if (limit.has_value()) {
      return {SafeClampInt64<IntT>(limit->first),
              SafeClampInt64<IntT>(limit->second)};
    }
    return {std::numeric_limits<IntT>::lowest(),
            std::numeric_limits<IntT>::max()};
  }

  int64_t min_64 = default_min;
  int64_t max_64 = default_max;

  if (limit.has_value()) {
    bool lower_unconstrained =
        (limit->first == std::numeric_limits<int64_t>::min());
    bool upper_unconstrained =
        (limit->second == std::numeric_limits<int64_t>::max());

    if (lower_unconstrained && upper_unconstrained) {
      min_64 = default_min;
      max_64 = default_max;
    } else if (lower_unconstrained) {
      max_64 = limit->second;
      min_64 =
          (max_64 < default_min) ? max_64 - default_range_size : default_min;
    } else if (upper_unconstrained) {
      min_64 = limit->first;
      max_64 =
          (min_64 > default_max) ? min_64 + default_range_size : default_max;
    } else {
      min_64 = limit->first;
      max_64 = limit->second;
    }
  }

  return {SafeClampInt64<IntT>(min_64), SafeClampInt64<IntT>(max_64)};
}

// Samples an integer offset in [0, span_minus_1] using strictly
// `steps_per_element` transitions of `engine` (1 draw for spans <= 2^20, 2
// draws for spans <= 2^32, and 4 draws for 64-bit spans). Using a fixed draw
// count instead of rejection sampling in absl::uniform_int_distribution avoids
// per-element rejection loops on sub-byte types (such as s4/u4) and guarantees
// exact partition-count invariance across worker threads.
uint64_t SampleIntegralOffset(std::minstd_rand0* engine, uint64_t span_minus_1,
                              int64_t steps_per_element) {
  if (steps_per_element == 1) {
    const uint64_t draw0 = DrawMinstdZeroBased(engine);
    return (draw0 * (span_minus_1 + 1)) / kMinstdSpan;
  }
  if (steps_per_element == 2) {
    const uint64_t draw0 = DrawMinstdZeroBased(engine);
    const uint64_t draw1 = DrawMinstdZeroBased(engine);
    const uint64_t draw62 = draw0 * kMinstdSpan + draw1;
    return static_cast<uint64_t>(
        (static_cast<absl::uint128>(draw62) * (span_minus_1 + 1)) /
        kMinstdSpanSquared);
  }
  const uint64_t draw0 = DrawMinstdZeroBased(engine);
  const uint64_t draw1 = DrawMinstdZeroBased(engine);
  const uint64_t draw2 = DrawMinstdZeroBased(engine);
  const uint64_t draw3 = DrawMinstdZeroBased(engine);
  const uint64_t u64 =
      (static_cast<uint64_t>(UniformUint32From2Draws(draw0, draw1)) << 32) |
      UniformUint32From2Draws(draw2, draw3);
  if (span_minus_1 == std::numeric_limits<uint64_t>::max()) {
    return u64;
  }
  return static_cast<uint64_t>(
      (static_cast<absl::uint128>(u64) *
       (static_cast<absl::uint128>(span_minus_1) + 1)) >>
      64);
}

template <typename IntT>
void PopulateWithRandomIntegralDataWithBounds(
    Literal* literal, std::minstd_rand0* engine, bool no_duplicates, IntT min,
    IntT max, std::optional<IntT> alignment,
    std::optional<IntT> index_known_zeroes) {
  CHECK_NOTNULL(engine);
  CHECK_EQ(literal->shape().element_type(),
           primitive_util::NativeToPrimitiveType<IntT>());
  CHECK(!(no_duplicates && alignment.has_value()))
      << "Cannot set both no_duplicates and alignment for integral types.";
  if (no_duplicates &&
      ShapeUtil::ElementsIn(literal->shape()) < static_cast<int64_t>(max)) {
    int32_t start = 0;
    if (primitive_util::IsSignedIntegralType(literal->shape().element_type())) {
      // Generate negative numbers also.
      auto size = literal->data<IntT>().size();
      start = size / 2 - size;
    }
    std::iota(literal->data<IntT>().begin(), literal->data<IntT>().end(),
              static_cast<IntT>(start));
    std::shuffle(literal->data<IntT>().begin(), literal->data<IntT>().end(),
                 *engine);
  } else {
    absl::Span<IntT> data = literal->data<IntT>();
    if (static_cast<int64_t>(data.size()) < kMinElementsPerPartition) {
      absl::uniform_int_distribution<RngT<IntT>> generator(
          static_cast<RngT<IntT>>(min), static_cast<RngT<IntT>>(max));
      for (IntT& value : data) {
        value = static_cast<IntT>(generator(*engine));
        if (alignment.has_value()) {
          value = (value / *alignment) * *alignment;
        }
        if (index_known_zeroes.has_value()) {
          value &= ~*index_known_zeroes;
        }
      }
      return;
    }
    // Map signed bounds into an order-preserving unsigned domain by flipping
    // the sign bit so `u_max - u_min` never overflows signed arithmetic.
    using SignedOrUnsignedRngT = RngT<IntT>;
    using UnsignedRngT = std::make_unsigned_t<SignedOrUnsignedRngT>;
    constexpr UnsignedRngT kSignBit =
        std::is_signed_v<SignedOrUnsignedRngT>
            ? (UnsignedRngT{1} << (sizeof(UnsignedRngT) * 8 - 1))
            : UnsignedRngT{0};
    const SignedOrUnsignedRngT min_rng = static_cast<SignedOrUnsignedRngT>(min);
    const SignedOrUnsignedRngT max_rng = static_cast<SignedOrUnsignedRngT>(max);
    const uint64_t u_min =
        static_cast<uint64_t>(static_cast<UnsignedRngT>(min_rng) ^ kSignBit);
    const uint64_t u_max =
        static_cast<uint64_t>(static_cast<UnsignedRngT>(max_rng) ^ kSignBit);
    CHECK_LE(u_min, u_max);
    const uint64_t span_minus_1 = u_max - u_min;

    const int64_t steps_per_element =
        span_minus_1 < (uint64_t{1} << 20)
            ? 1
            : (span_minus_1 <= std::numeric_limits<uint32_t>::max() ? 2 : 4);
    ParallelPopulateWithEngine(
        static_cast<int64_t>(data.size()), engine, steps_per_element,
        [&](int64_t start, int64_t end, std::minstd_rand0* local_engine) {
          for (int64_t i = start; i < end; ++i) {
            const uint64_t offset = SampleIntegralOffset(
                local_engine, span_minus_1, steps_per_element);
            const SignedOrUnsignedRngT sampled =
                static_cast<SignedOrUnsignedRngT>(
                    static_cast<UnsignedRngT>(u_min + offset) ^ kSignBit);
            IntT value = static_cast<IntT>(sampled);
            if (alignment.has_value()) {
              value = (value / *alignment) * *alignment;
            }
            if (index_known_zeroes.has_value()) {
              value &= ~*index_known_zeroes;
            }
            data[i] = value;
          }
        });
  }
}

// Populates a PRED (boolean) literal with uniform random booleans, using the
// parallel fixed-step engine path for tensors of at least
// kMinElementsPerPartition elements.
void PopulateWithRandomPredData(Literal* literal, std::minstd_rand0* engine) {
  CHECK_NOTNULL(engine);
  absl::Span<bool> data = literal->data<bool>();
  if (static_cast<int64_t>(data.size()) < kMinElementsPerPartition) {
    absl::uniform_int_distribution<int> generator(0, 1);
    for (bool& value : data) {
      value = generator(*engine);
    }
    return;
  }
  constexpr std::minstd_rand0::result_type kHalfMax =
      std::minstd_rand0::max() / 2;
  ParallelPopulateWithEngine(
      static_cast<int64_t>(data.size()), engine,
      /*steps_per_element=*/1,
      [&](int64_t start, int64_t end, std::minstd_rand0* local_engine) {
        for (int64_t i = start; i < end; ++i) {
          data[i] = (*local_engine)() > kHalfMax;
        }
      });
}

}  // namespace

/* static */ Literal LiteralUtil::CreateFromDimensions(
    PrimitiveType primitive_type, absl::Span<const int64_t> dimensions) {
  return Literal::CreateFromShape(
      ShapeUtil::MakeShape(primitive_type, dimensions));
}

/* static */ Literal LiteralUtil::ConvertF8E4M3FNToF32(
    const LiteralSlice& f8e4m3fn_literal) {
  return ConvertType<tsl::float8_e4m3fn, float>(f8e4m3fn_literal);
}

/* static */ Literal LiteralUtil::ConvertF8E5M2ToF32(
    const LiteralSlice& f8e5m2_literal) {
  return ConvertType<tsl::float8_e5m2, float>(f8e5m2_literal);
}

/* static */ Literal LiteralUtil::ConvertS8ToF32(
    const LiteralSlice& s8_literal) {
  return ConvertType<int8_t, float>(s8_literal);
}

/* static */ Literal LiteralUtil::ConvertBF16ToF32(
    const LiteralSlice& bf16_literal) {
  return ConvertType<bfloat16, float>(bf16_literal);
}

/* static */ Literal LiteralUtil::ConvertBF16ToF64(
    const LiteralSlice& bf16_literal) {
  return ConvertType<bfloat16, double>(bf16_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E4M3FNUZ(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e4m3fnuz>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E5M2FNUZ(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e5m2fnuz>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E5M2(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e5m2>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF8E4M3FN(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, tsl::float8_e4m3fn>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToBF16(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, bfloat16>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToS8(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, int8_t>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF32ToF64(
    const LiteralSlice& f32_literal) {
  return ConvertType<float, double>(f32_literal);
}

/* static */ Literal LiteralUtil::ConvertF64ToBF16(
    const LiteralSlice& f64_literal) {
  return ConvertType<double, bfloat16>(f64_literal);
}

/* static */ Literal LiteralUtil::ConvertF64ToF32(
    const LiteralSlice& f64_literal) {
  return ConvertType<double, float>(f64_literal);
}

/* static */ Literal LiteralUtil::ConvertS32ToF32(
    const LiteralSlice& s32_literal) {
  return ConvertType<int32_t, float>(s32_literal);
}

/* static */ Literal LiteralUtil::ConvertS32ToS1(
    const LiteralSlice& s32_literal) {
  return ConvertType<int32_t, tsl::int1>(s32_literal);
}

/* static */ Literal LiteralUtil::CreateToken() {
  return Literal(ShapeUtil::MakeTokenShape());
}

/* static */ Literal LiteralUtil::Zero(PrimitiveType primitive_type) {
  return CreateScalar<ZeroProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::One(PrimitiveType primitive_type) {
  return CreateScalar<OneProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::MinValue(PrimitiveType primitive_type) {
  return CreateScalar<MinProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::MaxValue(PrimitiveType primitive_type) {
  return CreateScalar<MaxProvider>(primitive_type);
}

/* static */ Literal LiteralUtil::MaxFiniteValue(PrimitiveType primitive_type) {
  return CreateScalar<MaxFiniteProvider>(primitive_type);
}

/* static */ absl::StatusOr<Literal> LiteralUtil::NanValue(
    PrimitiveType primitive_type) {
  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          return LiteralUtil::CreateR0<NativeT>(
              std::numeric_limits<NativeT>::quiet_NaN());
        }
        if constexpr (primitive_util::IsComplexType(primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          auto nan =
              std::numeric_limits<typename NativeT::value_type>::quiet_NaN();
          return LiteralUtil::CreateR0<NativeT>(NativeT(nan, nan));
        }
        return InvalidArgument("Invalid type for NanValue: %s",
                               PrimitiveType_Name(primitive_type));
      },
      primitive_type);
}

/* static */ Literal LiteralUtil::CreateR1(const tsl::core::Bitmap& values) {
  Literal literal(
      ShapeUtil::MakeShape(PRED, {static_cast<int64_t>(values.bits())}));
  literal.PopulateR1(values);
  return literal;
}

/* static */ Literal LiteralUtil::CreateR1U8(absl::string_view value) {
  Literal literal(
      ShapeUtil::MakeShape(U8, {static_cast<int64_t>(value.size())}));
  for (int i = 0, end = value.size(); i < end; ++i) {
    literal.Set<uint8_t>({i}, value[i]);
  }
  return literal;
}

/* static */ Literal LiteralUtil::CreateR2F32Linspace(float from, float to,
                                                      int64_t rows,
                                                      int64_t cols) {
  auto value = MakeLinspaceArray2D(from, to, rows, cols);
  return CreateR2FromArray2D(*value);
}

/* static */ Literal LiteralUtil::ReshapeSlice(
    absl::Span<const int64_t> new_dimensions,
    absl::Span<const int64_t> minor_to_major, const LiteralSlice& literal) {
  int64_t new_num_elements = 1;
  for (int64_t i = 0, end = new_dimensions.size(); i < end; ++i) {
    new_num_elements *= new_dimensions[i];
  }
  CHECK_EQ(ShapeUtil::ElementsIn(literal.shape()), new_num_elements);
  CHECK_EQ(new_dimensions.size(), minor_to_major.size());

  Literal new_literal(
      ShapeUtil::MakeShape(literal.shape().element_type(), new_dimensions));

  // Create a new shape with the given minor-to-major layout. This shape is used
  // solely for converting linear address to multi-dimensional addresses when
  // writing elements to the new literal.
  Shape shape_with_layout = new_literal.shape();
  *shape_with_layout.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);

  // Copy data into new literal, element-by-element.
  const int64_t num_elements = ShapeUtil::ElementsIn(literal.shape());
  primitive_util::ByteWidthTypeSwitch(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        const auto* src_data =
            static_cast<const NativeT*>(literal.untyped_data());
        auto* dst_data = static_cast<NativeT*>(new_literal.untyped_data());
        for (int64_t i = 0; i < num_elements; ++i) {
          auto from_multi_index =
              IndexUtil::LinearIndexToMultidimensionalIndex(literal.shape(), i);
          auto to_multi_index = IndexUtil::LinearIndexToMultidimensionalIndex(
              shape_with_layout, i);
          dst_data[IndexUtil::MultidimensionalIndexToLinearIndex(
              new_literal.shape(), to_multi_index)] =
              src_data[IndexUtil::MultidimensionalIndexToLinearIndex(
                  literal.shape(), from_multi_index)];
        }
      },
      literal.shape().element_type());

  return new_literal;
}

/* static */ Literal LiteralUtil::GetFirstScalarLiteral(
    const LiteralSlice& literal) {
  CHECK(literal.shape().IsArray());
  CHECK_GT(ShapeUtil::ElementsIn(literal.shape()), 0);
  Literal scalar(ShapeUtil::MakeScalarShape(literal.shape().element_type()));
  primitive_util::ByteWidthTypeSwitch(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        *static_cast<NativeT*>(scalar.untyped_data()) =
            *static_cast<const NativeT*>(literal.untyped_data());
      },
      literal.shape().element_type());
  return scalar;
}

/*static*/ Literal LiteralUtil::GetScalarLiteral(
    const LiteralBase& literal, absl::Span<const int64_t> multi_index) {
  CHECK(literal.shape().IsArray());
  Literal scalar(ShapeUtil::MakeScalarShape(literal.shape().element_type()));
  const int64_t linear_index = IndexUtil::MultidimensionalIndexToLinearIndex(
      literal.shape(), multi_index);
  primitive_util::ByteWidthTypeSwitch(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        *static_cast<NativeT*>(scalar.untyped_data()) =
            static_cast<const NativeT*>(literal.untyped_data())[linear_index];
      },
      literal.shape().element_type());
  return scalar;
}

/*static*/ void LiteralUtil::SetScalarLiteral(
    MutableLiteralBase& literal, absl::Span<const int64_t> multi_index,
    const LiteralBase& scalar) {
  CHECK(literal.shape().IsArray());
  CHECK(ShapeUtil::IsScalar(scalar.shape()));
  CHECK_EQ(literal.shape().element_type(), scalar.shape().element_type());
  const int64_t linear_index = IndexUtil::MultidimensionalIndexToLinearIndex(
      literal.shape(), multi_index);
  primitive_util::ByteWidthTypeSwitch(
      [&](auto primitive_type_constant) -> void {
        using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
        static_cast<NativeT*>(literal.untyped_data())[linear_index] =
            *static_cast<const NativeT*>(scalar.untyped_data());
      },
      literal.shape().element_type());
}

/* static */ Literal LiteralUtil::MaxElement(const LiteralSlice& literal) {
  CHECK(literal.shape().IsArray());
  CHECK_GT(ShapeUtil::ElementsIn(literal.shape()), 0);
  return CreateScalar<MaxElementProvider>(literal.shape().element_type(),
                                          literal);
}

/* static */ Literal LiteralUtil::MakeTuple(
    absl::Span<const Literal* const> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto* element : elements) {
    element_shapes.push_back(&element->shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    CHECK_OK(literal.CopyFrom(*elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleFromSlices(
    absl::Span<const LiteralSlice> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int i = 0, end = elements.size(); i < end; ++i) {
    CHECK_OK(literal.CopyFrom(elements[i], /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ Literal LiteralUtil::MakeTupleOwned(
    std::vector<Literal> elements) {
  std::vector<const Shape*> element_shapes;
  element_shapes.reserve(elements.size());
  for (const auto& element : elements) {
    element_shapes.push_back(&element.shape());
  }
  Literal literal(ShapeUtil::MakeTupleShapeWithPtrs(element_shapes));
  for (int64_t i = 0, end = elements.size(); i < end; ++i) {
    CHECK_OK(
        literal.MoveFrom(std::move(elements[i]), /*dest_shape_index=*/{i}));
  }
  return literal;
}

/* static */ std::string LiteralUtil::MultiIndexAsString(
    absl::Span<const int64_t> multi_index) {
  return StrCat("{", absl::StrJoin(multi_index, ","), "}");
}

/* static */ std::optional<int64_t> LiteralUtil::LiteralAsScalarInt64(
    const Literal& l) {
  if (!ShapeUtil::IsEffectiveScalar(l.shape())) {
    VLOG(2) << "literal is not an effective scalar: " << l.ToString();
    return std::nullopt;
  }
  return l.GetFirstInteger();
}

absl::StatusOr<Literal> MakeFakeLiteral(const Shape& shape, bool pseudo_random,
                                        bool use_large_range) {
  auto engine = pseudo_random ? std::make_unique<std::minstd_rand0>() : nullptr;
  return MakeFakeLiteral(shape, engine.get(), /*limit=*/std::nullopt,
                         /*is_sorted=*/false,
                         /*no_duplicates=*/false, use_large_range,
                         /*max_bits_of_precision=*/std::nullopt,
                         /*index_alignment=*/std::nullopt,
                         /*index_known_zeroes=*/std::nullopt,
                         /*float_generator=*/nullptr,
                         /*interval=*/std::nullopt);
}

absl::StatusOr<Literal> MakeFakeLiteral(
    const Shape& shape, std::minstd_rand0* engine,
    std::optional<std::pair<int64_t, int64_t>> limit, bool is_sorted,
    bool no_duplicates, bool use_large_range,
    std::optional<int64_t> max_bits_of_precision,
    std::optional<int64_t> index_alignment,
    std::optional<uint64_t> index_known_zeroes,
    std::function<double(std::minstd_rand0*)> float_generator,
    std::optional<ConstraintInterval> interval) {
  if (shape.IsTuple()) {
    std::vector<Literal> elements;
    const auto& shape_tuple_shapes = shape.tuple_shapes();
    elements.reserve(shape_tuple_shapes.size());
    for (const Shape& element_shape : shape_tuple_shapes) {
      ABSL_ASSIGN_OR_RETURN(
          Literal element,
          MakeFakeLiteral(element_shape, engine, limit, is_sorted,
                          no_duplicates, use_large_range, max_bits_of_precision,
                          index_alignment, index_known_zeroes, float_generator,
                          interval));
      elements.push_back(std::move(element));
    }
    return LiteralUtil::MakeTupleOwned(std::move(elements));
  }
  if (engine == nullptr) {
    return Literal::CreateFromShape(shape);
  }
  // Clear tiles/element size in shape's layout before using it for creating
  // literal. Ensure a default layout is present first before mutating layout
  // fields.
  Shape new_shape = shape;
  if (!new_shape.has_layout()) {
    LayoutUtil::SetToDefaultLayout(&new_shape);
  }
  new_shape.mutable_layout()->clear_tiles();
  new_shape.mutable_layout()->set_tail_padding_alignment_in_elements(1);
  new_shape.mutable_layout()->set_element_size_in_bits(0);
  Literal literal(new_shape);

  ABSL_RETURN_IF_ERROR(primitive_util::PrimitiveTypeSwitch<absl::Status>(
      [&](auto primitive_type_constant) -> absl::Status {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          if constexpr (primitive_util::IsFloatingPointType(
                            primitive_type_constant)) {
            PopulateWithFloatingPointData<NativeT>(
                &literal, engine, no_duplicates, use_large_range,
                max_bits_of_precision, float_generator, interval);
            return absl::OkStatus();
          }
          if constexpr (primitive_type_constant == PRED) {
            PopulateWithRandomPredData(&literal, engine);
            return absl::OkStatus();
          }
          if constexpr (primitive_util::IsIntegralType(
                            primitive_type_constant)) {
            auto [min, max] = GetIntegralBounds<NativeT>(
                new_shape, use_large_range, no_duplicates, limit);
            if (max_bits_of_precision.has_value()) {
              max = std::min(max,
                             static_cast<NativeT>(1 << *max_bits_of_precision));
              if (primitive_util::IsSignedIntegralType(
                      primitive_type_constant)) {
                min = std::max(
                    min, static_cast<NativeT>(-(1 << *max_bits_of_precision)));
              }
            }
            std::optional<NativeT> alignment_native = std::nullopt;
            if (index_alignment.has_value()) {
              alignment_native = static_cast<NativeT>(*index_alignment);
            }
            std::optional<NativeT> index_known_zeroes_native = std::nullopt;
            if (index_known_zeroes.has_value()) {
              index_known_zeroes_native =
                  static_cast<NativeT>(*index_known_zeroes);
            }
            PopulateWithRandomIntegralDataWithBounds<NativeT>(
                &literal, engine, no_duplicates, min, max, alignment_native,
                index_known_zeroes_native);
            if (is_sorted) {
              std::sort(literal.data<NativeT>().begin(),
                        literal.data<NativeT>().end());
            }
            return absl::OkStatus();
          }
          if constexpr (primitive_util::IsComplexType(
                            primitive_type_constant)) {
            PopulateWithComplexData<NativeT>(&literal, engine, no_duplicates,
                                             use_large_range, float_generator);
            return absl::OkStatus();
          }
        }
        return Unimplemented(
            "Unsupported type for fake random literal generation with bounds: "
            "%s",
            ShapeUtil::HumanString(shape));
      },
      shape.element_type()));
  return std::move(literal);
}

/*static*/ std::vector<const Literal*> LiteralUtil::MakePointers(
    absl::Span<const Literal> literals) {
  std::vector<const Literal*> pointers(literals.size());
  for (int i = 0; i < literals.size(); i++) {
    pointers[i] = &literals[i];
  }
  return pointers;
}

absl::StatusOr<Literal> MaterializeSparseOperand(
    const LiteralSlice& values, const LiteralSlice& indices,
    const SparsityConfig::TensorSparsityConfig& config) {
  int64_t dim = config.dimension();
  int64_t block_size = config.block_size();
  TF_RET_CHECK(config.num_non_zero() == 1)
      << "Only 1:N sparsity is currently supported.";
  TF_RET_CHECK(config.stride() == 1)
      << "Strided sparsity is not supported yet.";
  TF_RET_CHECK(dim >= 0 && dim < values.shape().dimensions().size())
      << "Invalid sparsity dimension: " << dim << ", must be in [0, "
      << values.shape().dimensions().size() << ")";
  TF_RET_CHECK(block_size > 0)
      << "block_size must be strictly positive in SparsityConfig.";
  TF_RET_CHECK(ShapeUtil::SameDimensions(values.shape(), indices.shape()))
      << "Values shape " << ShapeUtil::HumanString(values.shape())
      << " must have the same dimensions as indices shape "
      << ShapeUtil::HumanString(indices.shape());
  TF_RET_CHECK(ShapeUtil::ElementIsIntegral(indices.shape()))
      << "Indices literal must be an integral type.";

  std::vector<int64_t> dense_dims(values.shape().dimensions().begin(),
                                  values.shape().dimensions().end());
  TF_RET_CHECK(std::numeric_limits<int64_t>::max() / block_size >=
               dense_dims[dim])
      << "Dense dimension overflow triggered by large block_size.";
  dense_dims[dim] = dense_dims[dim] * block_size;
  Literal dense = Literal::CreateFromShape(
      ShapeUtil::MakeShape(values.shape().element_type(), dense_dims));

  return primitive_util::PrimitiveTypeSwitch<absl::StatusOr<Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<Literal> {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = typename primitive_util::PrimitiveTypeToNative<
              primitive_type_constant>::type;
          absl::Status status = absl::OkStatus();
          values.EachCell<NativeT>([&](absl::Span<const int64_t> sparse_index,
                                       NativeT val) {
            if (!status.ok()) {
              return;
            }
            std::optional<int64_t> position_in_block_opt =
                indices.GetIntegralAsS64(sparse_index);
            if (!position_in_block_opt.has_value()) {
              status = absl::InvalidArgumentError(
                  absl::StrCat("Failed to get integral value for indices at ",
                               LiteralUtil::MultiIndexAsString(sparse_index)));
              return;
            }
            int64_t position_in_block = *position_in_block_opt;
            if (position_in_block < 0 || position_in_block >= block_size) {
              status = absl::InvalidArgumentError(
                  absl::StrCat("Invalid position in block: ", position_in_block,
                               ", must be in [0, ", block_size, ")"));
              return;
            }
            absl::InlinedVector<int64_t, 8> out_index(sparse_index.begin(),
                                                      sparse_index.end());
            out_index[dim] = sparse_index[dim] * block_size + position_in_block;
            dense.Set<NativeT>(out_index, val);
          });
          if (!status.ok()) {
            return status;
          }
          return std::move(dense);
        }
        return absl::InvalidArgumentError(
            "Unsupported type for sparse matrix materialization.");
      },
      values.shape().element_type());
}

}  // namespace xla
