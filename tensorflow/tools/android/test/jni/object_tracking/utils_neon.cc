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

// NEON implementations of Image methods for compatible devices.  Control
// should never enter this compilation unit on incompatible devices.

#ifdef __ARM_NEON

#include <arm_neon.h>

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

inline static float GetSum(const float32x4_t& values) {
  static float32_t summed_values[4];
  vst1q_f32(summed_values, values);
  return summed_values[0]
       + summed_values[1]
       + summed_values[2]
       + summed_values[3];
}


float ComputeMeanNeon(const float* const values, const int num_vals) {
  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals = (const float32_t* const) values;
  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    accum = vaddq_f32(accum, vld1q_f32(&arm_vals[offset]));
  }

  // Pull the accumulated values into a single variable.
  float sum = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    sum += values[offset];
  }

  const float mean_neon = sum / static_cast<float>(num_vals);

#ifdef SANITY_CHECKS
  const float mean_cpu = ComputeMeanCpu(values, num_vals);
  SCHECK(NearlyEqual(mean_neon, mean_cpu, EPSILON * num_vals),
        "Neon mismatch with CPU mean! %.10f vs %.10f",
        mean_neon, mean_cpu);
#endif

  return mean_neon;
}


float ComputeStdDevNeon(const float* const values,
                        const int num_vals, const float mean) {
  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals = (const float32_t* const) values;
  const float32x4_t mean_vec = vdupq_n_f32(-mean);

  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    const float32x4_t deltas =
        vaddq_f32(mean_vec, vld1q_f32(&arm_vals[offset]));

    accum = vmlaq_f32(accum, deltas, deltas);
  }

  // Pull the accumulated values into a single variable.
  float squared_sum = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    squared_sum += Square(values[offset] - mean);
  }

  const float std_dev_neon = sqrt(squared_sum / static_cast<float>(num_vals));

#ifdef SANITY_CHECKS
  const float std_dev_cpu = ComputeStdDevCpu(values, num_vals, mean);
  SCHECK(NearlyEqual(std_dev_neon, std_dev_cpu, EPSILON * num_vals),
        "Neon mismatch with CPU std dev! %.10f vs %.10f",
        std_dev_neon, std_dev_cpu);
#endif

  return std_dev_neon;
}


float ComputeCrossCorrelationNeon(const float* const values1,
                                  const float* const values2,
                                  const int num_vals) {
  SCHECK(num_vals >= 8, "Not enough values to merit NEON: %d", num_vals);

  const float32_t* const arm_vals1 = (const float32_t* const) values1;
  const float32_t* const arm_vals2 = (const float32_t* const) values2;

  float32x4_t accum = vdupq_n_f32(0.0f);

  int offset = 0;
  for (; offset <= num_vals - 4; offset += 4) {
    accum = vmlaq_f32(accum,
                      vld1q_f32(&arm_vals1[offset]),
                      vld1q_f32(&arm_vals2[offset]));
  }

  // Pull the accumulated values into a single variable.
  float sxy = GetSum(accum);

  // Get the remaining 1 to 3 values.
  for (; offset < num_vals; ++offset) {
    sxy += values1[offset] * values2[offset];
  }

  const float cross_correlation_neon = sxy / num_vals;

#ifdef SANITY_CHECKS
  const float cross_correlation_cpu =
      ComputeCrossCorrelationCpu(values1, values2, num_vals);
  SCHECK(NearlyEqual(cross_correlation_neon, cross_correlation_cpu,
                    EPSILON * num_vals),
        "Neon mismatch with CPU cross correlation! %.10f vs %.10f",
        cross_correlation_neon, cross_correlation_cpu);
#endif

  return cross_correlation_neon;
}

}  // namespace tf_tracking

#endif  // __ARM_NEON
