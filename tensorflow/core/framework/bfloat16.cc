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

#include "tensorflow/core/framework/bfloat16.h"

#include "Eigen/Core"  // from @eigen_archive

namespace tensorflow {

void RoundFloatToBFloat16(const float* src, bfloat16* dst, int64_t size) {
  Eigen::Map<const Eigen::ArrayXf> src_eigen(src, size);
  Eigen::Map<Eigen::Array<bfloat16, Eigen::Dynamic, 1>> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<bfloat16>();
}

void FloatToBFloat16(const float* src, bfloat16* dst, int64_t size) {
  for (; size != 0; src++, dst++, size--) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(dst, src, sizeof(bfloat16));
#else
    memcpy(
        dst,
        reinterpret_cast<const char*>(src) + sizeof(float) - sizeof(bfloat16),
        sizeof(bfloat16));
#endif
  }
}

void BFloat16ToFloat(const bfloat16* src, float* dst, int64_t size) {
  Eigen::Map<const Eigen::Array<bfloat16, Eigen::Dynamic, 1>> src_eigen(src,
                                                                        size);
  Eigen::Map<Eigen::ArrayXf> dst_eigen(dst, size);
  dst_eigen = src_eigen.cast<float>();
}

}  // end namespace tensorflow
