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

#include "tensorflow/core/framework/tensor_testutil.h"
#include <cmath>

namespace tensorflow {
namespace test {

template <typename T>
void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  const T* Tx = x.flat<T>().data();
  const T* Ty = y.flat<T>().data();
  const auto size = x.NumElements();

  // Tolerance's type (RealType) can be different from T.
  // For example, if T = std::complex<float>, then RealType = float.
  // Did not use std::numeric_limits<T> because
  // 1) It returns 0 for Eigen::half.
  // 2) It doesn't support T=std::complex<RealType>.
  //    (Would have to write a templated struct to handle this.)
  typedef decltype(Eigen::NumTraits<T>::epsilon()) RealType;
  const RealType kSlackFactor = static_cast<RealType>(5.0);
  const RealType kDefaultTol = kSlackFactor * Eigen::NumTraits<T>::epsilon();
  const RealType typed_atol =
      (atol < 0) ? kDefaultTol : static_cast<RealType>(atol);
  const RealType typed_rtol =
      (rtol < 0) ? kDefaultTol : static_cast<RealType>(rtol);
  ASSERT_GE(typed_atol, static_cast<RealType>(0.0))
      << "typed_atol is negative: " << typed_atol;
  ASSERT_GE(typed_rtol, static_cast<RealType>(0.0))
      << "typed_rtol is negative: " << typed_rtol;
  const int max_failures = 10;
  int num_failures = 0;
  for (int i = 0; i < size; ++i) {
    EXPECT_TRUE(
        internal::Helper<T>::IsClose(Tx[i], Ty[i], typed_atol, typed_rtol))
        << "index = " << (++num_failures, i) << " x = " << Tx[i]
        << " y = " << Ty[i] << " typed_atol = " << typed_atol
        << " typed_rtol = " << typed_rtol;
    ASSERT_LT(num_failures, max_failures) << "Too many mismatches, giving up.";
  }
}

void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  internal::AssertSameTypeDims(x, y);
  switch (x.dtype()) {
    case DT_HALF:
      ExpectClose<Eigen::half>(x, y, atol, rtol);
      break;
    case DT_FLOAT:
      ExpectClose<float>(x, y, atol, rtol);
      break;
    case DT_DOUBLE:
      ExpectClose<double>(x, y, atol, rtol);
      break;
    case DT_COMPLEX64:
      ExpectClose<complex64>(x, y, atol, rtol);
      break;
    case DT_COMPLEX128:
      ExpectClose<complex128>(x, y, atol, rtol);
      break;
    default:
      LOG(FATAL) << "Unexpected type : " << DataTypeString(x.dtype());
  }
}

}  // end namespace test
}  // end namespace tensorflow
