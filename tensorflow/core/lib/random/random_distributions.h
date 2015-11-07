#ifndef TENSORFLOW_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_
#define TENSORFLOW_LIB_RANDOM_RANDOM_DISTRIBUTIONS_H_

#include <math.h>
#include <string.h>
#include <algorithm>

#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {
namespace random {

// Helper function to convert a 32-bit integer to a float between [0..1).
PHILOX_DEVICE_INLINE float Uint32ToFloat(uint32 x);
// Helper function to convert two 32-bit integers to a double between [0..1).
PHILOX_DEVICE_INLINE double Uint64ToDouble(uint32 x0, uint32 x1);

// A class that generates uniform distribution random numbers from the
// underlying random integer generator.
// Arguments:
//   Generator: a generator type that returns a number of uint32 upon each
//              each invocation. It needs to define kResultElementCount for the
//              sample count for each invocation, and ResultType for actual
//              returned sample type.
//   RealType: the data type of the real numberes that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class Generator, typename RealType>
class UniformDistribution;

template <class Generator>
class UniformDistribution<Generator, float> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = Generator::kResultElementCount;
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

 private:
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
//   RealType: the data type of the real numberes that will be returned by the
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

template <class Generator>
class NormalDistribution<Generator, float> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount = Generator::kResultElementCount;
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
//   RealType: the data type of the real numberes that will be returned by the
//             distribution. This could be either float or double for now.
// This class is meant to be implemented through specialization. The default
// is not defined by design.
template <class SingleSampleGenerator, typename RealType>
class TruncatedNormalDistribution;

// Partial specialization for float.
template <class SingleSampleGenerator>
class TruncatedNormalDistribution<SingleSampleGenerator, float> {
 public:
  // The number of elements that will be returned.
  static const int kResultElementCount =
      SingleSampleGenerator::kNativeElementCount;
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
        if (fabs(f[i]) < kTruncateValue) {
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
        if (fabs(d[i]) < kTruncateValue) {
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
  const float u2 = sqrt(-2.0f * log(u1));
#if defined(__linux)
  sincosf(v1, f0, f1);
#else
  *f0 = sinf(v1);
  *f1 = cosf(v1);
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
  const double u2 = sqrt(-2.0 * log(u1));
#if defined(__linux)
  sincos(v1, d0, d1);
#else
  *d0 = sin(v1);
  *d1 = cos(v1);
#endif
  *d0 *= u2;
  *d1 *= u2;
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
