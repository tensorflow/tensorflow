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
// identical values.
template <typename T>
void ExpectTensorEqual(const Tensor& x, const Tensor& y);

// Expects "x" and "y" are tensors of the same type, same shape, and
// approximate equal values, each within "abs_err".
template <typename T>
void ExpectTensorNear(const Tensor& x, const Tensor& y, const T& abs_err);

// Expects "x" and "y" are tensors of the same type (float or double),
// same shape and element-wise difference between x and y is no more
// than atol + rtol * abs(x). If atol or rtol is negative, it is replaced
// with a default tolerance value = data type's epsilon * kSlackFactor.
void ExpectClose(const Tensor& x, const Tensor& y, double atol = -1.0,
                 double rtol = -1.0);

// Implementation details.

namespace internal {

template <typename T>
struct is_floating_point_type {
  static constexpr bool value = std::is_same<T, Eigen::half>::value ||
                                std::is_same<T, float>::value ||
                                std::is_same<T, double>::value ||
                                std::is_same<T, std::complex<float>>::value ||
                                std::is_same<T, std::complex<double>>::value;
};

template <typename T>
inline void ExpectEqual(const T& a, const T& b) {
  EXPECT_EQ(a, b);
}

template <>
inline void ExpectEqual<float>(const float& a, const float& b) {
  EXPECT_FLOAT_EQ(a, b);
}

template <>
inline void ExpectEqual<double>(const double& a, const double& b) {
  EXPECT_DOUBLE_EQ(a, b);
}

template <>
inline void ExpectEqual<complex64>(const complex64& a, const complex64& b) {
  EXPECT_FLOAT_EQ(a.real(), b.real()) << a << " vs. " << b;
  EXPECT_FLOAT_EQ(a.imag(), b.imag()) << a << " vs. " << b;
}

template <>
inline void ExpectEqual<complex128>(const complex128& a, const complex128& b) {
  EXPECT_DOUBLE_EQ(a.real(), b.real()) << a << " vs. " << b;
  EXPECT_DOUBLE_EQ(a.imag(), b.imag()) << a << " vs. " << b;
}

template <typename T>
inline void ExpectEqual(const T& a, const T& b, int index) {
  EXPECT_EQ(a, b) << " at index " << index;
}

template <>
inline void ExpectEqual<float>(const float& a, const float& b, int index) {
  EXPECT_FLOAT_EQ(a, b) << " at index " << index;
}

template <>
inline void ExpectEqual<double>(const double& a, const double& b, int index) {
  EXPECT_DOUBLE_EQ(a, b) << " at index " << index;
}

template <>
inline void ExpectEqual<complex64>(const complex64& a, const complex64& b,
                                   int index) {
  EXPECT_FLOAT_EQ(a.real(), b.real())
      << a << " vs. " << b << " at index " << index;
  EXPECT_FLOAT_EQ(a.imag(), b.imag())
      << a << " vs. " << b << " at index " << index;
}

template <>
inline void ExpectEqual<complex128>(const complex128& a, const complex128& b,
                                    int index) {
  EXPECT_DOUBLE_EQ(a.real(), b.real())
      << a << " vs. " << b << " at index " << index;
  EXPECT_DOUBLE_EQ(a.imag(), b.imag())
      << a << " vs. " << b << " at index " << index;
}

inline void AssertSameTypeDims(const Tensor& x, const Tensor& y) {
  ASSERT_EQ(x.dtype(), y.dtype());
  ASSERT_TRUE(x.IsSameSize(y))
      << "x.shape [" << x.shape().DebugString() << "] vs "
      << "y.shape [ " << y.shape().DebugString() << "]";
}

template <typename T, bool is_fp = is_floating_point_type<T>::value>
struct Expector;

template <typename T>
struct Expector<T, false> {
  static void Equal(const T& a, const T& b) { ExpectEqual(a, b); }

  static void Equal(const Tensor& x, const Tensor& y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<T>::v());
    AssertSameTypeDims(x, y);
    const auto size = x.NumElements();
    const T* a = x.unaligned_flat<T>().data();
    const T* b = y.unaligned_flat<T>().data();
    for (int i = 0; i < size; ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }
};

// Partial specialization for float and double.
template <typename T>
struct Expector<T, true> {
  static void Equal(const T& a, const T& b) { ExpectEqual(a, b); }

  static void Equal(const Tensor& x, const Tensor& y) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<T>::v());
    AssertSameTypeDims(x, y);
    const auto size = x.NumElements();
    const T* a = x.unaligned_flat<T>().data();
    const T* b = y.unaligned_flat<T>().data();
    for (int i = 0; i < size; ++i) {
      ExpectEqual(a[i], b[i]);
    }
  }

  static bool Near(const T& a, const T& b, const double abs_err) {
    // Need a == b so that infinities are close to themselves.
    return (a == b) ||
           (static_cast<double>(Eigen::numext::abs(a - b)) <= abs_err);
  }

  static void Near(const Tensor& x, const Tensor& y, const double abs_err) {
    ASSERT_EQ(x.dtype(), DataTypeToEnum<T>::v());
    AssertSameTypeDims(x, y);
    const auto size = x.NumElements();
    const T* a = x.unaligned_flat<T>().data();
    const T* b = y.unaligned_flat<T>().data();
    for (int i = 0; i < size; ++i) {
      EXPECT_TRUE(Near(a[i], b[i], abs_err))
          << "a = " << a[i] << " b = " << b[i] << " index = " << i;
    }
  }
};

template <typename T>
struct Helper {
  // Assumes atol and rtol are nonnegative.
  static bool IsClose(const T& x, const T& y, const T& atol, const T& rtol) {
    // Need x == y so that infinities are close to themselves.
    return (x == y) ||
           (Eigen::numext::abs(x - y) <= atol + rtol * Eigen::numext::abs(x));
  }
};

template <typename T>
struct Helper<std::complex<T>> {
  static bool IsClose(const std::complex<T>& x, const std::complex<T>& y,
                      const T& atol, const T& rtol) {
    return Helper<T>::IsClose(x.real(), y.real(), atol, rtol) &&
           Helper<T>::IsClose(x.imag(), y.imag(), atol, rtol);
  }
};

}  // namespace internal

template <typename T>
void ExpectTensorEqual(const Tensor& x, const Tensor& y) {
  internal::Expector<T>::Equal(x, y);
}

template <typename T>
void ExpectTensorNear(const Tensor& x, const Tensor& y, const double abs_err) {
  static_assert(internal::is_floating_point_type<T>::value,
                "T is not a floating point types.");
  ASSERT_GE(abs_err, 0.0) << "abs_error is negative" << abs_err;
  internal::Expector<T>::Near(x, y, abs_err);
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_TESTUTIL_H_
