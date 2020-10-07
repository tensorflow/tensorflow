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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_

#include <numeric>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace test {

// Constructs a scalar tensor with 'val'.
template <typename T>
Tensor AsScalar(const T& val) {
  Tensor ret(DataTypeToEnum<T>::value, {});
  ret.scalar<T>()() = val;
  return ret;
}

// Constructs a flat tensor with 'vals'.
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals) {
  Tensor ret(DataTypeToEnum<T>::value, {static_cast<int64>(vals.size())});
  std::copy_n(vals.data(), vals.size(), ret.flat<T>().data());
  return ret;
}

// Constructs a tensor of "shape" with values "vals".
template <typename T>
Tensor AsTensor(gtl::ArraySlice<T> vals, const TensorShape& shape) {
  Tensor ret;
  CHECK(ret.CopyFrom(AsTensor(vals), shape));
  return ret;
}

// Fills in '*tensor' with 'vals'. E.g.,
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillValues<float>(&x, {11, 21, 21, 22});
template <typename T>
void FillValues(Tensor* tensor, gtl::ArraySlice<T> vals) {
  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    std::copy_n(vals.data(), vals.size(), flat.data());
  }
}

// Fills in '*tensor' with 'vals', converting the types as needed.
template <typename T, typename SrcType>
void FillValues(Tensor* tensor, std::initializer_list<SrcType> vals) {
  auto flat = tensor->flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  if (flat.size() > 0) {
    size_t i = 0;
    for (auto itr = vals.begin(); itr != vals.end(); ++itr, ++i) {
      flat(i) = T(*itr);
    }
  }
}

// Fills in '*tensor' with a sequence of value of val, val+1, val+2, ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillIota<float>(&x, 1.0);
template <typename T>
void FillIota(Tensor* tensor, const T& val) {
  auto flat = tensor->flat<T>();
  std::iota(flat.data(), flat.data() + flat.size(), val);
}

// Fills in '*tensor' with a sequence of value of fn(0), fn(1), ...
//   Tensor x(&alloc, DT_FLOAT, TensorShape({2, 2}));
//   test::FillFn<float>(&x, [](int i)->float { return i*i; });
template <typename T>
void FillFn(Tensor* tensor, std::function<T(int)> fn) {
  auto flat = tensor->flat<T>();
  for (int i = 0; i < flat.size(); ++i) flat(i) = fn(i);
}

// Expects "x" and "y" are tensors of the same type, same shape, and
// identical values (within 4 ULPs for floating point types).
void ExpectEqual(const Tensor& x, const Tensor& y);

// Expects "x" and "y" are tensors of the same (floating point) type,
// same shape and element-wise difference between x and y is no more
// than atol + rtol * abs(x). If atol or rtol is negative, the data type's
// epsilon * kSlackFactor is used.
void ExpectClose(const Tensor& x, const Tensor& y, double atol = -1.0,
                 double rtol = -1.0);

// Expects "x" and "y" are tensors of the same type T, same shape, and
// equal values. Consider using ExpectEqual above instead.
template <typename T>
void ExpectTensorEqual(const Tensor& x, const Tensor& y) {
  EXPECT_EQ(x.dtype(), DataTypeToEnum<T>::value);
  ExpectEqual(x, y);
}

// Expects "x" and "y" are tensors of the same type T, same shape, and
// approximate equal values. Consider using ExpectClose above instead.
template <typename T>
void ExpectTensorNear(const Tensor& x, const Tensor& y, double atol) {
  EXPECT_EQ(x.dtype(), DataTypeToEnum<T>::value);
  ExpectClose(x, y, atol, /*rtol=*/0.0);
}

// For tensor_testutil_test only.
namespace internal_test {
::testing::AssertionResult IsClose(Eigen::half x, Eigen::half y,
                                   double atol = -1.0, double rtol = -1.0);
::testing::AssertionResult IsClose(float x, float y, double atol = -1.0,
                                   double rtol = -1.0);
::testing::AssertionResult IsClose(double x, double y, double atol = -1.0,
                                   double rtol = -1.0);
}  // namespace internal_test

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
