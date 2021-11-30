/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
#define TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions_utils.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace random {

// Helper function to convert a 16-bit integer to a half between [0..1).
PHILOX_DEVICE_INLINE Eigen::half Uint16ToHalf(uint16 x);
// Helper function to convert a 16-bit integer to a bfloat16 between [0..1).
PHILOX_DEVICE_INLINE bfloat16 Uint16ToGfloat16(uint16 x);

// Computes a + b. Requires that the result is representable in the destination
// type and that b is not maximal (i.e. b + 1 is not 0). Notably, the addend b
// need *not* be representable in that type. (The condition on b excludes the
// extremal case INT_MIN + UINT_MAX = INT_MAX, which this function cannot
// compute.)
template <typename Int>
PHILOX_DEVICE_INLINE Int SignedAdd(Int a,
                                   typename std::make_unsigned<Int>::type b) {
  // Implementation note: both b_div_2 and b - b_div_2 are positive and
  // representable as Int.
  auto b_div_2 = b >> 1;
  return a + static_cast<Int>(b_div_2) + static_cast<Int>(b - b_div_2);
}

// A class that generates uniform distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for the
//              actual returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType>
class UniformDistribution;

template <class Generator>
class UniformDistribution<Generator, Eigen::half> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<Eigen::half, kResultElementCount> ResultType;
  typedef Eigen::half ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint16ToHalf(sample[i]);  // Truncate the upper 16 bits.
    }
    return result;
  }
};

template <class Generator>
class UniformDistribution<Generator, bfloat16> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<bfloat16, kResultElementCount> ResultType;
  typedef bfloat16 ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint16ToGfloat16(sample[i]);
    }
    return result;
  }
};

template <class Generator>
class UniformDistribution<Generator, float> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<float, kResultElementCount> ResultType;
  typedef float ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint32ToFloat(sample[i]);
    }
    return result;
  }
};

template <class Generator>
class UniformDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<double, kResultElementCount> ResultType;
  typedef double ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = Uint64ToDouble(sample[2 * i], sample[2 * i + 1]);
    }
    return result;
  }
};

template <class Generator>
class UniformDistribution<Generator, int32> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<int32, kResultElementCount> ResultType;
  typedef int32 ResultElementType;

  // Must have lo < hi
  UniformDistribution(int32_t lo, int32_t hi)
      : lo_(lo), range_(static_cast<uint32>(hi) - static_cast<uint32>(lo)) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = SignedAdd(lo_, sample[i] % range_);
    }
    return result;
  }

 private:
  // Note that lo_ is intentionally signed while range_ is intentionally
  // unsigned.  This is because hi - lo can overflow signed integers if
  // lo < 0 < hi, but always fits in unsigned.
  int32 lo_;
  uint32 range_;
};

template <class Generator>
class UniformDistribution<Generator, int64_t> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<int64_t, kResultElementCount> ResultType;
  typedef int64_t ResultElementType;

  // Must have lo < hi
  UniformDistribution(int64_t lo, int64_t hi)
      : lo_(lo), range_(static_cast<uint64>(hi) - static_cast<uint64>(lo)) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      auto bits = sample[2 * i] | static_cast<uint64>(sample[2 * i + 1]) << 32;
      result[i] = SignedAdd(lo_, bits % range_);
    }
    return result;
  }

 private:
  // Note that lo_ is intentionally signed while range_ is intentionally
  // unsigned.  This is because hi - lo can overflow signed integers if
  // lo < 0 < hi, but always fits in unsigned.
  int64_t lo_;
  uint64 range_;
};

// Similar to `UniformDistribution`, except that instead of generating numbers
// in the range [low, high), it generates numbers covering the whole range of
// the integer type.
template <typename Generator, typename IntType>
class UniformFullIntDistribution;

template <typename Generator, typename IntType>
class UniformFullIntDistribution32 {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<IntType, kResultElementCount> ResultType;
  typedef IntType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = sample[i];
    }
    return result;
  }
};

template <typename Generator, typename IntType>
class UniformFullIntDistribution64 {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<IntType, kResultElementCount> ResultType;
  typedef IntType ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; ++i) {
      result[i] = sample[2 * i] | static_cast<uint64>(sample[2 * i + 1]) << 32;
    }
    return result;
  }
};

template <typename Generator>
class UniformFullIntDistribution<Generator, int32>
    : public UniformFullIntDistribution32<Generator, int32> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, uint32>
    : public UniformFullIntDistribution32<Generator, uint32> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, int64_t>
    : public UniformFullIntDistribution64<Generator, int64_t> {};
template <typename Generator>
class UniformFullIntDistribution<Generator, uint64>
    : public UniformFullIntDistribution64<Generator, uint64> {};

// A class that adapts the underlying native multiple samples to return a single
// sample at a time.
template <class Generator>
class SingleSampleAdapter {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = 1;
  // The number of elements that will be returned by the underlying generator.
  static constexpr int kNativeElementCount = Generator::kResultElementCount;
  typedef typename Generator::ResultElementType ResultType;
  typedef typename Generator::ResultElementType ResultElementType;

  PHILOX_DEVICE_INLINE
  explicit SingleSampleAdapter(Generator* gen)
      : generator_(gen), used_result_index_(Generator::kResultElementCount) {}

  PHILOX_DEVICE_INLINE
  ResultType operator()() {
    if (used_result_index_ == Generator::kResultElementCount) {
      unused_results_ = (*generator_)();
      used_result_index_ = 0;
    }

    return unused_results_[used_result_index_++];
  }

  PHILOX_DEVICE_INLINE
  void Skip(uint64 num_skips) {
    if (!num_skips) {
      return;
    }
    int num_unused_results = kNativeElementCount - used_result_index_;
    if (num_skips <= num_unused_results) {
      used_result_index_ += num_skips;
      return;
    }
    num_skips -= num_unused_results;
    used_result_index_ = kNativeElementCount;
    SkipFromGenerator(num_skips / kNativeElementCount);
    num_skips = num_skips % kNativeElementCount;
    if (num_skips) {
      unused_results_ = (*generator_)();
      used_result_index_ = num_skips;
    }
  }

 private:
  // This implementation iteratively skips over `num_skips` samples
  // from `generator_`. There is an O(1) implementation for PhiloxRandom
  // in random_distributions.cc.
  PHILOX_DEVICE_INLINE
  void SkipFromGenerator(uint64 num_skips) {
    while (num_skips--) {
      (*generator_)();
    }
  }

  Generator* generator_;
  typename Generator::ResultType unused_results_;
  int used_result_index_;
};

// A class that generates unit normal distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType>
class NormalDistribution;

PHILOX_DEVICE_INLINE
void BoxMullerDouble(uint32 x0, uint32 x1, uint32 x2, uint32 x3, double* d0,
                     double* d1);

// Exactly like the float version, except that we convert to half afterwards;
// since we don't have half-precision sin/cos even on GPUs, there's nothing to
// gain from working in half internally.
template <class Generator>
class NormalDistribution<Generator, Eigen::half> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<Eigen::half, kResultElementCount> ResultType;
  typedef Eigen::half ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; i += 2) {
      float f[2];
      BoxMullerFloat(sample[i], sample[i + 1], &f[0], &f[1]);
      result[i] = Eigen::half(f[0]);
      result[i + 1] = Eigen::half(f[1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, bfloat16> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<bfloat16, kResultElementCount> ResultType;
  typedef bfloat16 ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    static_assert(kResultElementCount % 2 == 0,
                  "kResultElementCount should be an even number");
    for (int i = 0; i < kResultElementCount; i += 2) {
      float f[2];
      // Box-Muller transform requires processing 2 elements at a time.
      BoxMullerFloat(sample[i], sample[i + 1], &f[0], &f[1]);
      result[i] = bfloat16(f[0]);
      result[i + 1] = bfloat16(f[1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, float> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<float, kResultElementCount> ResultType;
  typedef float ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; i += 2) {
      BoxMullerFloat(sample[i], sample[i + 1], &result[i], &result[i + 1]);
    }
    return result;
  }
};

template <class Generator>
class NormalDistribution<Generator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = false;
  typedef Array<double, kResultElementCount> ResultType;
  typedef double ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(Generator* gen) {
    typename Generator::ResultType sample = (*gen)();
    ResultType result;
    for (int i = 0; i < kResultElementCount; i += 2) {
      const int i2 = 2 * i;
      BoxMullerDouble(sample[i2], sample[i2 + 1], sample[i2 + 2],
                      sample[i2 + 3], &result[i], &result[i + 1]);
    }
    return result;
  }
};

// A class that returns standard normal distribution between
// [-kTruncateValue, kTruncateValue].
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numbers that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class SingleSampleGenerator, typename RealType>
class TruncatedNormalDistribution;

// Exactly like the float version, except that we convert to half afterwards;
// since we don't have half-precision sin/cos even on GPUs, there's nothing to
// gain from working in half internally.
template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, Eigen::half> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  // The threshold where the normal distribution is truncated.
  const float kTruncateValue = 2.0f;

  typedef Array<Eigen::half, kResultElementCount> ResultType;
  typedef Eigen::half ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      // Repeatedly take samples from the normal distribution, until we have
      // the desired number of elements that fall within the pre-defined cutoff
      // threshold.
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      float f[2];
      BoxMullerFloat(x0, x1, &f[0], &f[1]);

      if (Eigen::numext::abs(f[0]) < kTruncateValue) {
        results[index++] = Eigen::half(f[0]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(f[1]) < kTruncateValue) {
        results[index++] = Eigen::half(f[1]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, bfloat16> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  // The threshold where the normal distribution is truncated.
  const float kTruncateValue = 2.0f;

  typedef Array<bfloat16, kResultElementCount> ResultType;
  typedef bfloat16 ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      // Repeatedly take samples from the normal distribution, until we have
      // the desired number of elements that fall within the pre-defined cutoff
      // threshold.
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      float f[2];
      BoxMullerFloat(x0, x1, &f[0], &f[1]);

      if (Eigen::numext::abs(f[0]) < kTruncateValue) {
        results[index++] = bfloat16(f[0]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(f[1]) < kTruncateValue) {
        results[index++] = bfloat16(f[1]);
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

// Partial specialization for float.
template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, float> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  // The threshold where the normal distribution is truncated.
  const float kTruncateValue = 2.0f;

  typedef Array<float, kResultElementCount> ResultType;
  typedef float ResultElementType;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      // Repeatedly take samples from the normal distribution, until we have
      // the desired number of elements that fall within the pre-defined cutoff
      // threshold.
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      float f[2];
      BoxMullerFloat(x0, x1, &f[0], &f[1]);

      if (Eigen::numext::abs(f[0]) < kTruncateValue) {
        results[index++] = f[0];
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(f[1]) < kTruncateValue) {
        results[index++] = f[1];
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

// Partial specialization for double.
template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, double> {
 public:
  // The number of elements that will be returned.
  static constexpr int kResultElementCount =
      (SingleSampleGenerator::kNativeElementCount > 1)
          ? SingleSampleGenerator::kNativeElementCount / 2
          : 1;
  // Cost of generation of a single element (in cycles).
  static constexpr int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static constexpr bool kVariableSamplesPerOutput = true;
  typedef Array<double, kResultElementCount> ResultType;
  typedef double ResultElementType;
  const double kTruncateValue = 2.0;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (true) {
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      const uint32 x2 = (*gen)();
      const uint32 x3 = (*gen)();
      double d[2];
      BoxMullerDouble(x0, x1, x2, x3, &d[0], &d[1]);

      if (Eigen::numext::abs(d[0]) < kTruncateValue) {
        results[index++] = d[0];
        if (index >= kResultElementCount) {
          return results;
        }
      }
      if (Eigen::numext::abs(d[1]) < kTruncateValue) {
        results[index++] = d[1];
        if (index >= kResultElementCount) {
          return results;
        }
      }
    }
  }
};

// Helper function to convert four 32-bit uniform integers to two doubles
// under the unit normal distribution.
PHILOX_DEVICE_INLINE
void BoxMullerDouble(uint32 x0, uint32 x1, uint32 x2, uint32 x3, double* d0,
                     double* d1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const double epsilon = 1.0e-7;
  double u1 = Uint64ToDouble(x0, x1);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const double v1 = 2 * M_PI * Uint64ToDouble(x2, x3);
  const double u2 = Eigen::numext::sqrt(-2.0 * Eigen::numext::log(u1));
#if !defined(__linux__)
  *d0 = Eigen::numext::sin(v1);
  *d1 = Eigen::numext::cos(v1);
#else
  sincos(v1, d0, d1);
#endif
  *d0 *= u2;
  *d1 *= u2;
}

// Helper function to convert an 16-bit integer to a half between [0..1).
PHILOX_DEVICE_INLINE Eigen::half Uint16ToHalf(uint16 x) {
  // IEEE754 halfs are formatted as follows (MSB first):
  //    sign(1) exponent(5) mantissa(10)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 15  -- an excess 15 representation of a zero exponent
  //    mantissa == 10 random bits
  const uint16 man = x & 0x3ffu;  // 10 bit mantissa
  const uint16 exp = static_cast<uint16>(15);
  const uint16 val = (exp << 10) | man;

  Eigen::half result = Eigen::numext::bit_cast<Eigen::half>(val);
  return result - Eigen::half(1.0);
}

// Helper function to convert an 16-bit integer to a bfloat16 between [0..1).
// This can create a uniform distribution of values between [0..1).
PHILOX_DEVICE_INLINE bfloat16 Uint16ToGfloat16(uint16 x) {
  // bfloat are formatted as follows (MSB first):
  //    sign(1) exponent(8) mantissa(7)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 127  -- an excess 127 representation of a zero exponent
  //    mantissa == 7 random bits
  const uint16 man = x & 0x7fu;  // 7 bit mantissa
  const uint16 exp = static_cast<uint16>(127);
  const uint16 val = (exp << 7) | man;

  bfloat16 result;
  memcpy(&result, &val, sizeof(val));
  // The mantissa has an implicit leading 1, so the above code creates a value
  // in [1, 2). The minus will not cause a rounding that makes the result 1.
  // Instead it will just be close to 1.
  return result - bfloat16(1.0);
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
