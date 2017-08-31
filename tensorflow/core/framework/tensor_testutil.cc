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

#include <cmath>
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace test {

template <typename T>
bool IsClose(const T& x, const T& y, double atol, double rtol) {
  // Need x == y so that infinities are close to themselves
  return x == y || std::abs(x - y) < atol + rtol * std::abs(x);
}

template <typename T>
void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  auto Tx = x.flat<T>();
  auto Ty = y.flat<T>();
  for (int i = 0; i < Tx.size(); ++i) {
    if (!IsClose(Tx(i), Ty(i), atol, rtol)) {
      LOG(ERROR) << "x = " << x.DebugString();
      LOG(ERROR) << "y = " << y.DebugString();
      LOG(ERROR) << "atol = " << atol << " rtol = " << rtol
                 << " tol = " << atol + rtol * std::abs(Tx(i));
      EXPECT_TRUE(false) << i << "-th element is not close " << Tx(i) << " vs. "
                         << Ty(i);
    }
  }
}

void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  internal::AssertSameTypeDims(x, y);
  switch (x.dtype()) {
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
