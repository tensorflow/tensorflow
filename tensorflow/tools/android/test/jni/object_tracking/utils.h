/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_UTILS_H_
#define TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_UTILS_H_

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

#include <cmath>  // for std::abs(float)

#ifndef HAVE_CLOCK_GETTIME
// Use gettimeofday() instead of clock_gettime().
#include <sys/time.h>
#endif  // ifdef HAVE_CLOCK_GETTIME

#include "tensorflow/tools/android/test/jni/object_tracking/logging.h"

// TODO(andrewharp): clean up these macros to use the codebase statndard.

// A very small number, generally used as the tolerance for accumulated
// floating point errors in bounds-checks.
#define EPSILON 0.00001f

#define SAFE_DELETE(pointer) {\
  if ((pointer) != NULL) {\
    LOGV("Safe deleting pointer: %s", #pointer);\
    delete (pointer);\
    (pointer) = NULL;\
  } else {\
    LOGV("Pointer already null: %s", #pointer);\
  }\
}


#ifdef __GOOGLE__

#define CHECK_ALWAYS(condition, format, ...) {\
  CHECK(condition) << StringPrintf(format, ##__VA_ARGS__);\
}

#define SCHECK(condition, format, ...) {\
  DCHECK(condition) << StringPrintf(format, ##__VA_ARGS__);\
}

#else

#define CHECK_ALWAYS(condition, format, ...) {\
  if (!(condition)) {\
    LOGE("CHECK FAILED (%s): " format, #condition, ##__VA_ARGS__);\
    abort();\
  }\
}

#ifdef SANITY_CHECKS
#define SCHECK(condition, format, ...) {\
  CHECK_ALWAYS(condition, format, ##__VA_ARGS__);\
}
#else
#define SCHECK(condition, format, ...) {}
#endif  // SANITY_CHECKS

#endif  // __GOOGLE__


#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#endif

inline static int64_t CurrentThreadTimeNanos() {
#ifdef HAVE_CLOCK_GETTIME
  struct timespec tm;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tm);
  return tm.tv_sec * 1000000000LL + tm.tv_nsec;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000000 + tv.tv_usec * 1000;
#endif
}

inline static int64_t CurrentRealTimeMillis() {
#ifdef HAVE_CLOCK_GETTIME
  struct timespec tm;
  clock_gettime(CLOCK_MONOTONIC, &tm);
  return tm.tv_sec * 1000LL + tm.tv_nsec / 1000000LL;
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}


template<typename T>
inline static T Square(const T a) {
  return a * a;
}


template<typename T>
inline static T Clip(const T a, const T floor, const T ceil) {
  SCHECK(ceil >= floor, "Bounds mismatch!");
  return (a <= floor) ? floor : ((a >= ceil) ? ceil : a);
}


template<typename T>
inline static int Floor(const T a) {
  return static_cast<int>(a);
}


template<typename T>
inline static int Ceil(const T a) {
  return Floor(a) + 1;
}


template<typename T>
inline static bool InRange(const T a, const T min, const T max) {
  return (a >= min) && (a <= max);
}


inline static bool ValidIndex(const int a, const int max) {
  return (a >= 0) && (a < max);
}


inline bool NearlyEqual(const float a, const float b, const float tolerance) {
  return std::abs(a - b) < tolerance;
}


inline bool NearlyEqual(const float a, const float b) {
  return NearlyEqual(a, b, EPSILON);
}


template<typename T>
inline static int Round(const float a) {
  return (a - static_cast<float>(floor(a) > 0.5f) ? ceil(a) : floor(a));
}


template<typename T>
inline static void Swap(T* const a, T* const b) {
  // Cache out the VALUE of what's at a.
  T tmp = *a;
  *a = *b;

  *b = tmp;
}


static inline float randf() {
  return rand() / static_cast<float>(RAND_MAX);
}

static inline float randf(const float min_value, const float max_value) {
  return randf() * (max_value - min_value) + min_value;
}

static inline uint16_t RealToFixed115(const float real_number) {
  SCHECK(InRange(real_number, 0.0f, 2048.0f),
        "Value out of range! %.2f", real_number);

  static const float kMult = 32.0f;
  const float round_add = (real_number > 0.0f) ? 0.5f : -0.5f;
  return static_cast<uint16_t>(real_number * kMult + round_add);
}

static inline float FixedToFloat115(const uint16_t fp_number) {
  const float kDiv = 32.0f;
  return (static_cast<float>(fp_number) / kDiv);
}

static inline int RealToFixed1616(const float real_number) {
  static const float kMult = 65536.0f;
  SCHECK(InRange(real_number, -kMult, kMult),
        "Value out of range! %.2f", real_number);

  const float round_add = (real_number > 0.0f) ? 0.5f : -0.5f;
  return static_cast<int>(real_number * kMult + round_add);
}

static inline float FixedToFloat1616(const int fp_number) {
  const float kDiv = 65536.0f;
  return (static_cast<float>(fp_number) / kDiv);
}

template<typename T>
// produces numbers in range [0,2*M_PI] (rather than -PI,PI)
inline T FastAtan2(const T y, const T x) {
  static const T coeff_1 = (T)(M_PI / 4.0);
  static const T coeff_2 = (T)(3.0 * coeff_1);
  const T abs_y = fabs(y);
  T angle;
  if (x >= 0) {
    T r = (x - abs_y) / (x + abs_y);
    angle = coeff_1 - coeff_1 * r;
  } else {
    T r = (x + abs_y) / (abs_y - x);
    angle = coeff_2 - coeff_1 * r;
  }
  static const T PI_2 = 2.0 * M_PI;
  return y < 0 ? PI_2 - angle : angle;
}

#define NELEMS(X) (sizeof(X) / sizeof(X[0]))

namespace tf_tracking {

#ifdef __ARM_NEON
float ComputeMeanNeon(const float* const values, const int num_vals);

float ComputeStdDevNeon(const float* const values, const int num_vals,
                        const float mean);

float ComputeWeightedMeanNeon(const float* const values,
                              const float* const weights, const int num_vals);

float ComputeCrossCorrelationNeon(const float* const values1,
                                  const float* const values2,
                                  const int num_vals);
#endif

inline float ComputeMeanCpu(const float* const values, const int num_vals) {
  // Get mean.
  float sum = values[0];
  for (int i = 1; i < num_vals; ++i) {
    sum += values[i];
  }
  return sum / static_cast<float>(num_vals);
}


inline float ComputeMean(const float* const values, const int num_vals) {
  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeMeanNeon(values, num_vals) :
#endif
                      ComputeMeanCpu(values, num_vals);
}


inline float ComputeStdDevCpu(const float* const values,
                              const int num_vals,
                              const float mean) {
  // Get Std dev.
  float squared_sum = 0.0f;
  for (int i = 0; i < num_vals; ++i) {
    squared_sum += Square(values[i] - mean);
  }
  return sqrt(squared_sum / static_cast<float>(num_vals));
}


inline float ComputeStdDev(const float* const values,
                           const int num_vals,
                           const float mean) {
  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeStdDevNeon(values, num_vals, mean) :
#endif
                      ComputeStdDevCpu(values, num_vals, mean);
}


// TODO(andrewharp): Accelerate with NEON.
inline float ComputeWeightedMean(const float* const values,
                                 const float* const weights,
                                 const int num_vals) {
  float sum = 0.0f;
  float total_weight = 0.0f;
  for (int i = 0; i < num_vals; ++i) {
    sum += values[i] * weights[i];
    total_weight += weights[i];
  }
  return sum / num_vals;
}


inline float ComputeCrossCorrelationCpu(const float* const values1,
                                        const float* const values2,
                                        const int num_vals) {
  float sxy = 0.0f;
  for (int offset = 0; offset < num_vals; ++offset) {
    sxy += values1[offset] * values2[offset];
  }

  const float cross_correlation = sxy / num_vals;

  return cross_correlation;
}


inline float ComputeCrossCorrelation(const float* const values1,
                                     const float* const values2,
                                     const int num_vals) {
  return
#ifdef __ARM_NEON
      (num_vals >= 8) ? ComputeCrossCorrelationNeon(values1, values2, num_vals)
                      :
#endif
                      ComputeCrossCorrelationCpu(values1, values2, num_vals);
}


inline void NormalizeNumbers(float* const values, const int num_vals) {
  // Find the mean and then subtract so that the new mean is 0.0.
  const float mean = ComputeMean(values, num_vals);
  VLOG(2) << "Mean is " << mean;
  float* curr_data = values;
  for (int i = 0; i < num_vals; ++i) {
    *curr_data -= mean;
    curr_data++;
  }

  // Now divide by the std deviation so the new standard deviation is 1.0.
  // The numbers might all be identical (and thus shifted to 0.0 now),
  // so only scale by the standard deviation if this is not the case.
  const float std_dev = ComputeStdDev(values, num_vals, 0.0f);
  if (std_dev > 0.0f) {
    VLOG(2) << "Std dev is " << std_dev;
    curr_data = values;
    for (int i = 0; i < num_vals; ++i) {
      *curr_data /= std_dev;
      curr_data++;
    }
  }
}


// Returns the determinant of a 2x2 matrix.
template<class T>
inline T FindDeterminant2x2(const T* const a) {
  // Determinant: (ad - bc)
  return a[0] * a[3] - a[1] * a[2];
}


// Finds the inverse of a 2x2 matrix.
// Returns true upon success, false if the matrix is not invertible.
template<class T>
inline bool Invert2x2(const T* const a, float* const a_inv) {
  const float det = static_cast<float>(FindDeterminant2x2(a));
  if (fabs(det) < EPSILON) {
    return false;
  }
  const float inv_det = 1.0f / det;

  a_inv[0] = inv_det * static_cast<float>(a[3]);   // d
  a_inv[1] = inv_det * static_cast<float>(-a[1]);  // -b
  a_inv[2] = inv_det * static_cast<float>(-a[2]);  // -c
  a_inv[3] = inv_det * static_cast<float>(a[0]);   // a

  return true;
}

}  // namespace tf_tracking

#endif  // TENSORFLOW_TOOLS_ANDROID_TEST_JNI_OBJECT_TRACKING_UTILS_H_
