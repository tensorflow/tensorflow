/* Copyright 2015 Google Inc. All Rights Reserved.

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

namespace tensorflow {

void FloatToBFloat16(const float* src, bfloat16* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  for (; size; p += 2, q++, size--) {
    *q = p[1];
  }
}

void BFloat16ToFloat(const bfloat16* src, float* dst, int64 size) {
  const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
  uint16_t* q = reinterpret_cast<uint16_t*>(dst);
  for (; size; p++, q += 2, size--) {
    q[0] = 0;
    q[1] = *p;
  }
}

}  // end namespace tensorflow
