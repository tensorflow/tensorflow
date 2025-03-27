/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sparse_matmul_op_hwy.h"

#define HWY_DISABLED_TARGETS (HWY_AVX3)
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE \
  "third_party/tensorflow/core/kernels/sparse_matmul_op_hwy.cc"  // this file
#include "hwy/foreach_target.h"  // from @com_google_highway  // IWYU pragma: keep
#include "hwy/highway.h"  // from @com_google_highway

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

int GEPP_helper_hwy(
    const float** data3, int j, int end3,
    const std::vector<std::tuple<uint8, uint8, uint8, uint8>>& index3,
    float** out_ptrs, const float** right_ptrs) {
  using D = ScalableTag<float>;
  D d;
  for (; j + 1 < end3; j += 2) {
    auto [m, k1, k2, k3] = index3[j];
    auto [nm, nk1, nk2, nk3] = index3[j + 1];
    float* out = out_ptrs[m];
    float* nout = out_ptrs[nm];
    const auto* r1 = right_ptrs[k1];
    const auto* r2 = right_ptrs[k2];
    const auto* r3 = right_ptrs[k3];

    const auto* nr1 = right_ptrs[nk1];
    const auto* nr2 = right_ptrs[nk2];
    const auto* nr3 = right_ptrs[nk3];
    const auto l1 = Set(d, **data3);
    ++(*data3);
    const auto l2 = Set(d, **data3);
    ++(*data3);
    const auto l3 = Set(d, **data3);
    ++(*data3);
    const auto nl1 = Set(d, **data3);
    ++(*data3);
    const auto nl2 = Set(d, **data3);
    ++(*data3);
    const auto nl3 = Set(d, **data3);
    ++(*data3);
    const auto num_operands = Lanes(d);
    for (int k = 0; k < 128 / (8 * num_operands); ++k) {
      //  TwoMulAdd3Way(l1, l2, l3, r1, r2, r3, out);
      {
        auto c1 = Load(d, out);
        const auto b1 = Load(d, r1);
        const auto b2 = Load(d, r2);
        const auto b3 = Load(d, r3);

        auto c2 = Load(d, out + num_operands);
        const auto b4 = Load(d, r1 + num_operands);
        const auto b5 = Load(d, r2 + num_operands);
        const auto b6 = Load(d, r3 + num_operands);

        c1 = MulAdd(l1, b1, c1);
        c2 = MulAdd(l1, b4, c2);
        c1 = MulAdd(l2, b2, c1);
        c2 = MulAdd(l2, b5, c2);
        c1 = MulAdd(l3, b3, c1);
        c2 = MulAdd(l3, b6, c2);
        StoreU(c1, d, out);
        StoreU(c2, d, out + num_operands);
        out += 2 * num_operands;
        r1 += 2 * num_operands;
        r2 += 2 * num_operands;
        r3 += 2 * num_operands;
      }
      // TwoMulAdd3Way(l1, l2, l3, r1, r2, r3, out);
      {
        auto c1 = Load(d, out);
        const auto b1 = Load(d, r1);
        const auto b2 = Load(d, r2);
        const auto b3 = Load(d, r3);

        auto c2 = Load(d, out + num_operands);
        const auto b4 = Load(d, r1 + num_operands);
        const auto b5 = Load(d, r2 + num_operands);
        const auto b6 = Load(d, r3 + num_operands);

        c1 = MulAdd(l1, b1, c1);
        c2 = MulAdd(l1, b4, c2);
        c1 = MulAdd(l2, b2, c1);
        c2 = MulAdd(l2, b5, c2);
        c1 = MulAdd(l3, b3, c1);
        c2 = MulAdd(l3, b6, c2);
        StoreU(c1, d, out);
        StoreU(c2, d, out + num_operands);
        out += 2 * num_operands;
        r1 += 2 * num_operands;
        r2 += 2 * num_operands;
        r3 += 2 * num_operands;
      }
    }
    for (int k = 0; k < 128 / (8 * num_operands); ++k) {
      // TwoMulAdd3Way(nl1, nl2, nl3, nr1, nr2, nr3, nout);
      {
        auto c1 = Load(d, nout);
        const auto b1 = Load(d, nr1);
        const auto b2 = Load(d, nr2);
        const auto b3 = Load(d, nr3);

        auto c2 = Load(d, nout + num_operands);
        const auto b4 = Load(d, nr1 + num_operands);
        const auto b5 = Load(d, nr2 + num_operands);
        const auto b6 = Load(d, nr3 + num_operands);

        c1 = MulAdd(nl1, b1, c1);
        c2 = MulAdd(nl1, b4, c2);
        c1 = MulAdd(nl2, b2, c1);
        c2 = MulAdd(nl2, b5, c2);
        c1 = MulAdd(nl3, b3, c1);
        c2 = MulAdd(nl3, b6, c2);
        StoreU(c1, d, nout);
        StoreU(c2, d, nout + num_operands);
        nout += 2 * num_operands;
        nr1 += 2 * num_operands;
        nr2 += 2 * num_operands;
        nr3 += 2 * num_operands;
      }
      // TwoMulAdd3Way(nl1, nl2, nl3, nr1, nr2, nr3, nout);
      {
        auto c1 = Load(d, nout);
        const auto b1 = Load(d, nr1);
        const auto b2 = Load(d, nr2);
        const auto b3 = Load(d, nr3);

        auto c2 = Load(d, nout + num_operands);
        const auto b4 = Load(d, nr1 + num_operands);
        const auto b5 = Load(d, nr2 + num_operands);
        const auto b6 = Load(d, nr3 + num_operands);

        c1 = MulAdd(nl1, b1, c1);
        c2 = MulAdd(nl1, b4, c2);
        c1 = MulAdd(nl2, b2, c1);
        c2 = MulAdd(nl2, b5, c2);
        c1 = MulAdd(nl3, b3, c1);
        c2 = MulAdd(nl3, b6, c2);
        StoreU(c1, d, nout);
        StoreU(c2, d, nout + num_operands);
        nout += 2 * num_operands;
        nr1 += 2 * num_operands;
        nr2 += 2 * num_operands;
        nr3 += 2 * num_operands;
      }
    }
  }
  return j;
}

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace hwy {
HWY_EXPORT(GEPP_helper_hwy);

int GEPP_helper(
    const float** data3, int j, int end3,
    const std::vector<std::tuple<uint8, uint8, uint8, uint8>>& index3,
    float** out_ptrs, const float** right_ptrs) {
  return HWY_DYNAMIC_DISPATCH(GEPP_helper_hwy)(data3, j, end3, index3, out_ptrs,
                                               right_ptrs);
}
}  // namespace hwy
#endif  // HWY_ONCE
