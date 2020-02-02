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

#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/bfloat16.h"

namespace tensorflow {

void FloatToBFloat16(const float* src, bfloat16* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p += 2, q++, size--) {
    *q = p[0];
  }
#else
  for (; size != 0; p += 2, q++, size--) {
    *q = p[1];
  }
#endif
}

void FloatToBFloat16(const Eigen::ThreadPoolDevice& d, const float* src,
                     bfloat16* dst, int64 size) {
  auto ParallelFloatToBFloat16 = [&](Eigen::Index start, Eigen::Index end) {
    FloatToBFloat16(src + start, dst + start, end - start);
  };
  const int input_bytes = size * sizeof(float);
  const int output_bytes = size * sizeof(bfloat16);
  const int compute_cycles = (Eigen::TensorOpCost::AddCost<uint16_t>() * 3);
  const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
  d.parallelFor(size, cost, ParallelFloatToBFloat16);
}

void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (; size != 0; p++, q += 2, size--) {
    q[0] = *p;
    q[1] = 0;
  }
#else
  for (; size != 0; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
#endif
}

void BFloat16ToFloat(const Eigen::ThreadPoolDevice& d, const bfloat16* src,
                     float* dst, int64 size) {
  auto ParallelBFloat16ToFloat = [&](Eigen::Index start, Eigen::Index end) {
    BFloat16ToFloat(src + start, dst + start, end - start);
  };
  const int input_bytes = size * sizeof(bfloat16);
  const int output_bytes = size * sizeof(float);
  const int compute_cycles = (Eigen::TensorOpCost::AddCost<uint16_t>() * 3);
  const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
  d.parallelFor(size, cost, ParallelBFloat16ToFloat);
}

}  // end namespace tensorflow
