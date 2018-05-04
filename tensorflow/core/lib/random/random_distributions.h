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

#ifndef TENSORFLOW_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
#define TENSORFLOW_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_

#define _USE_MATH_DEFINES
#include <math.h>
#include <cmath>
#undef _USE_MATH_DEFINES

#include <string.h>
#include <algorithm>
#include <type_traits>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/lib/bfloat16/bfloat16.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {
namespace random {

// Helper function to convert a 16-bit integer to a half between [0..1).
PHILOX_DEVICE_INLINE Eigen::half Uint16ToHalf(uint16 x);
// Helper function to convert a 16-bit integer to a bfloat16 between [0..1).
PHILOX_DEVICE_INLINE bfloat16 Uint16ToGfloat16(uint16 x);
// Helper function to convert a 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32 x);
// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ToDouble(uint32 x0, uint32 x1);

// Computes a + b. Requires that the result is representable in the destination
// type and that b is not maximal (i.e. b + 1 is not 0). Notably, the addend b
// need *not* be representable in that type. (The condition on b excludes the
// extremal case INT_MIN + UINT_MAX = INT_MAX, which this function cannot
// compute.)
template <typename Int>
PHILOX_DEVICE_INLINE Int SignedAdd(Int a,
                                   typename std::make_unsigned<Int>::type b) {
  // Implementation note: both b_div_2 and b - b_div_2 are positive and
  // representatble as Int.
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
  typedef Array<int32, kResultElementCount> ResultType;
  typedef int32 ResultElementType;

  // Must have lo < hi
  UniformDistribution(int32 lo, int32 hi)
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
class UniformDistribution<Generator, int64> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 3;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
  typedef Array<int64, kResultElementCount> ResultType;
  typedef int64 ResultElementType;

  // Must have lo < hi
  UniformDistribution(int64 lo, int64 hi)
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
  int64 lo_;
  uint64 range_;
};

// A class that adapts the underlying native multiple samples to return a single
// sample at a time.
template <class Generator>
class SingleSampleAdapter {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = 1;
  // The number of elements that will be returned by the underlying generator.
  static const int kNativeElementCount = Generator::kResultElementCount;
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
void BoxMullerFloat(uint32 x0, uint32 x1, float* f0, float* f1);

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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount = Generator::kResultElementCount / 2;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 70;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = false;
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
  static const int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = true;
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

      for (int i = 0; i < 2; ++i) {
        if (Eigen::numext::abs(f[i]) < kTruncateValue) {
          results[index++] = Eigen::half(f[i]);
          if (index >= kResultElementCount) {
            return results;
          }
        }
      }
    }
  }
};

template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, bfloat16> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = true;
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

      for (int i = 0; i < 2; ++i) {
        if (Eigen::numext::abs(f[i]) < kTruncateValue) {
          results[index++] = bfloat16(f[i]);
          if (index >= kResultElementCount) {
            return results;
          }
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
  static const int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = true;
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

      for (int i = 0; i < 2; ++i) {
        if (Eigen::numext::abs(f[i]) < kTruncateValue) {
          results[index++] = f[i];
          if (index >= kResultElementCount) {
            return results;
          }
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
  static const int kResultElementCount =
      (SingleSampleGenerator::kNativeElementCount > 1)
          ? SingleSampleGenerator::kNativeElementCount / 2
          : 1;
  // Cost of generation of a single element (in cycles).
  static const int kElementCost = 90;
  // Indicate that this distribution may take variable number of samples
  // during the runtime.
  static const bool kVariableSamplesPerOutput = true;
  typedef Array<double, kResultElementCount> ResultType;
  typedef double ResultElementType;
  const double kTruncateValue = 2.0;

  PHILOX_DEVICE_INLINE
  ResultType operator()(SingleSampleGenerator* gen) {
    ResultType results;
    int index = 0;
    while (1) {
      const uint32 x0 = (*gen)();
      const uint32 x1 = (*gen)();
      const uint32 x2 = (*gen)();
      const uint32 x3 = (*gen)();
      double d[2];
      BoxMullerDouble(x0, x1, x2, x3, &d[0], &d[1]);

      for (int i = 0; i < 2; ++i) {
        if (Eigen::numext::abs(d[i]) < kTruncateValue) {
          results[index++] = d[i];
          if (index >= kResultElementCount) {
            return results;
          }
        }
      }
    }
  }
};

// Helper function to convert two 32-bit uniform integers to two floats
// under the unit normal distribution.
PHILOX_DEVICE_INLINE
void BoxMullerFloat(uint32 x0, uint32 x1, float* f0, float* f1) {
  // This function implements the Box-Muller transform:
  // http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Basic_form
  // Do not send a really small number to log().
  // We cannot mark "epsilon" as "static const" because NVCC would complain
  const float epsilon = 1.0e-7f;
  float u1 = Uint32ToFloat(x0);
  if (u1 < epsilon) {
    u1 = epsilon;
  }
  const float v1 = 2.0f * M_PI * Uint32ToFloat(x1);
  const float u2 = Eigen::numext::sqrt(-2.0f * Eigen::numext::log(u1));
#if defined(TENSORFLOW_USE_SYCL) || !defined(__linux__)
  *f0 = Eigen::numext::sin(v1);
  *f1 = Eigen::numext::cos(v1);
#else
  sincosf(v1, f0, f1);
#endif
  *f0 *= u2;
  *f1 *= u2;
}

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
#if defined(TENSORFLOW_USE_SYCL) || !defined(__linux__)
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

  Eigen::half result;
  result.x = val;
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

// Helper function to convert an 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32 x) {
  // IEEE754 floats are formatted as follows (MSB first):
  //    sign(1) exponent(8) mantissa(23)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 127  -- an excess 127 representation of a zero exponent
  //    mantissa == 23 random bits
  const uint32 man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32 exp = static_cast<uint32>(127);
  const uint32 val = (exp << 23) | man;

  // Assumes that endian-ness is same for float and uint32.
  float result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0f;
}

// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ToDouble(uint32 x0, uint32 x1) {
  // IEEE754 doubles are formatted as follows (MSB first):
  //    sign(1) exponent(11) mantissa(52)
  // Conceptually construct the following:
  //    sign == 0
  //    exponent == 1023  -- an excess 1023 representation of a zero exponent
  //    mantissa == 52 random bits
  const uint32 mhi = x0 & 0xfffffu;  // upper 20 bits of mantissa
  const uint32 mlo = x1;             // lower 32 bits of mantissa
  const uint64 man = (static_cast<uint64>(mhi) << 32) | mlo;  // mantissa
  const uint64 exp = static_cast<uint64>(1023);
  const uint64 val = (exp << 52) | man;
  // Assumes that endian-ness is same for double and uint64.
  double result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0;
}

}  // namespace random
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
