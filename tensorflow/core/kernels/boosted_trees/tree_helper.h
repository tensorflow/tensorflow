/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
#define TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
#include <cmath>

namespace tensorflow {

static bool GainsAreEqual(const float g1, const float g2) {
  const float kTolerance = 1e-15;
  return std::abs(g1 - g2) < kTolerance;
}

static bool GainIsLarger(const float g1, const float g2) {
  const float kTolerance = 1e-15;
  return g1 - g2 >= kTolerance;
}

static void CalculateWeightsAndGains(const float g, const float h,
                                     const float l1, const float l2,
                                     float* weight, float* gain) {
  const float kEps = 1e-15;
  // The formula for weight is -(g+l1*sgn(w))/(H+l2), for gain it is
  // (g+l1*sgn(w))^2/(h+l2).
  // This is because for each leaf we optimize
  // 1/2(h+l2)*w^2+g*w+l1*abs(w)
  float g_with_l1 = g;
  // Apply L1 regularization.
  // 1) Assume w>0 => w=-(g+l1)/(h+l2)=> g+l1 < 0 => g < -l1
  // 2) Assume w<0 => w=-(g-l1)/(h+l2)=> g-l1 > 0 => g > l1
  // For g from (-l1, l1), thus there is no solution => set to 0.
  if (l1 > 0) {
    if (g > l1) {
      g_with_l1 -= l1;
    } else if (g < -l1) {
      g_with_l1 += l1;
    } else {
      *weight = 0.0;
      *gain = 0.0;
      return;
    }
  }
  // Apply L2 regularization.
  if (h + l2 <= kEps) {
    // Avoid division by 0 or infinitesimal.
    *weight = 0;
    *gain = 0;
  } else {
    *weight = -g_with_l1 / (h + l2);
    *gain = -g_with_l1 * (*weight);
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BOOSTED_TREES_TREE_HELPER_H_
