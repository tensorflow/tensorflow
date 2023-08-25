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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {

::testing::AssertionResult IsSameType(const Tensor& x, const Tensor& y) {
  if (x.dtype() != y.dtype()) {
    return ::testing::AssertionFailure()
           << "Tensors have different dtypes (" << x.dtype() << " vs "
           << y.dtype() << ")";
  }
  return ::testing::AssertionSuccess();
}

::testing::AssertionResult IsSameShape(const Tensor& x, const Tensor& y) {
  if (!x.IsSameSize(y)) {
    return ::testing::AssertionFailure()
           << "Tensors have different shapes (" << x.shape().DebugString()
           << " vs " << y.shape().DebugString() << ")";
  }
  return ::testing::AssertionSuccess();
}

template <typename T>
static ::testing::AssertionResult EqualFailure(const T& x, const T& y) {
  return ::testing::AssertionFailure()
         << std::setprecision(std::numeric_limits<T>::digits10 + 2) << x
         << " not equal to " << y;
}

template <>
::testing::AssertionResult EqualFailure<int8>(const int8& x, const int8& y) {
  return EqualFailure(static_cast<int>(x), static_cast<int>(y));
}

static ::testing::AssertionResult IsEqual(float x, float y, Tolerance t) {
  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (t == Tolerance::kNone) {
    if (x == y) return ::testing::AssertionSuccess();
  } else {
    if (::testing::internal::CmpHelperFloatingPointEQ<float>("", "", x, y))
      return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
static ::testing::AssertionResult IsEqual(double x, double y, Tolerance t) {
  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (t == Tolerance::kNone) {
    if (x == y) return ::testing::AssertionSuccess();
  } else {
    if (::testing::internal::CmpHelperFloatingPointEQ<double>("", "", x, y))
      return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
static ::testing::AssertionResult IsEqual(Eigen::half x, Eigen::half y,
                                          Tolerance t) {
  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();

  // Below is a reimplementation of CmpHelperFloatingPointEQ<Eigen::half>, which
  // we cannot use because Eigen::half is not default-constructible.

  if (Eigen::numext::isnan(x) || Eigen::numext::isnan(y))
    return EqualFailure(x, y);

  auto sign_and_magnitude_to_biased = [](uint16_t sam) {
    const uint16_t kSignBitMask = 0x8000;
    if (kSignBitMask & sam) return ~sam + 1;  // negative number.
    return kSignBitMask | sam;                // positive number.
  };

  auto xb = sign_and_magnitude_to_biased(Eigen::numext::bit_cast<uint16_t>(x));
  auto yb = sign_and_magnitude_to_biased(Eigen::numext::bit_cast<uint16_t>(y));
  if (t == Tolerance::kNone) {
    if (xb == yb) return ::testing::AssertionSuccess();
  } else {
    auto distance = xb >= yb ? xb - yb : yb - xb;
    const uint16_t kMaxUlps = 4;
    if (distance <= kMaxUlps) return ::testing::AssertionSuccess();
  }
  return EqualFailure(x, y);
}
template <typename T>
static ::testing::AssertionResult IsEqual(const T& x, const T& y, Tolerance t) {
  if (::testing::internal::CmpHelperEQ<T>("", "", x, y))
    return ::testing::AssertionSuccess();
  return EqualFailure(x, y);
}

template <typename T>
static ::testing::AssertionResult IsEqual(const std::complex<T>& x,
                                          const std::complex<T>& y,
                                          Tolerance t) {
  if (IsEqual(x.real(), y.real(), t) && IsEqual(x.imag(), y.imag(), t))
    return ::testing::AssertionSuccess();
  return EqualFailure(x, y);
}

template <typename T>
static void ExpectEqual(const Tensor& x, const Tensor& y,
                        Tolerance t = Tolerance::kDefault) {
  const T* Tx = x.unaligned_flat<T>().data();
  const T* Ty = y.unaligned_flat<T>().data();
  auto size = x.NumElements();
  int max_failures = 10;
  int num_failures = 0;
  for (decltype(size) i = 0; i < size; ++i) {
    EXPECT_TRUE(IsEqual(Tx[i], Ty[i], t)) << "i = " << (++num_failures, i);
    ASSERT_LT(num_failures, max_failures) << "Too many mismatches, giving up.";
  }
}

template <typename T>
static ::testing::AssertionResult IsClose(const T& x, const T& y, const T& atol,
                                          const T& rtol) {
  // We consider NaNs equal for testing.
  if (Eigen::numext::isnan(x) && Eigen::numext::isnan(y))
    return ::testing::AssertionSuccess();
  if (x == y) return ::testing::AssertionSuccess();  // Handle infinity.
  auto tolerance = atol + rtol * Eigen::numext::abs(x);
  if (Eigen::numext::abs(x - y) <= tolerance)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << x << " not close to " << y;
}

template <typename T>
static ::testing::AssertionResult IsClose(const std::complex<T>& x,
                                          const std::complex<T>& y,
                                          const T& atol, const T& rtol) {
  if (IsClose(x.real(), y.real(), atol, rtol) &&
      IsClose(x.imag(), y.imag(), atol, rtol))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << x << " not close to " << y;
}

// Return type can be different from T, e.g. float for T=std::complex<float>.
template <typename T>
static auto GetTolerance(double tolerance) {
  using Real = typename Eigen::NumTraits<T>::Real;
  auto default_tol = static_cast<Real>(5.0) * Eigen::NumTraits<T>::epsilon();
  auto result = tolerance < 0.0 ? default_tol : static_cast<Real>(tolerance);
  EXPECT_GE(result, static_cast<Real>(0));
  return result;
}

template <typename T>
static void ExpectClose(const Tensor& x, const Tensor& y, double atol,
                        double rtol) {
  auto typed_atol = GetTolerance<T>(atol);
  auto typed_rtol = GetTolerance<T>(rtol);

  const T* Tx = x.unaligned_flat<T>().data();
  const T* Ty = y.unaligned_flat<T>().data();
  auto size = x.NumElements();
  int max_failures = 10;
  int num_failures = 0;
  for (decltype(size) i = 0; i < size; ++i) {
    EXPECT_TRUE(IsClose(Tx[i], Ty[i], typed_atol, typed_rtol))
        << "i = " << (++num_failures, i) << " Tx[i] = " << Tx[i]
        << " Ty[i] = " << Ty[i];
    ASSERT_LT(num_failures, max_failures)
        << "Too many mismatches (atol = " << atol << " rtol = " << rtol
        << "), giving up.";
  }
  EXPECT_EQ(num_failures, 0)
      << "Mismatches detected (atol = " << atol << " rtol = " << rtol << ").";
}

void ExpectEqual(const Tensor& x, const Tensor& y, Tolerance t) {
  ASSERT_TRUE(IsSameType(x, y));
  ASSERT_TRUE(IsSameShape(x, y));

  switch (x.dtype()) {
    case DT_FLOAT:
      return ExpectEqual<float>(x, y, t);
    case DT_DOUBLE:
      return ExpectEqual<double>(x, y, t);
    case DT_INT32:
      return ExpectEqual<int32>(x, y);
    case DT_UINT32:
      return ExpectEqual<uint32>(x, y);
    case DT_UINT16:
      return ExpectEqual<uint16>(x, y);
    case DT_UINT8:
      return ExpectEqual<uint8>(x, y);
    case DT_INT16:
      return ExpectEqual<int16>(x, y);
    case DT_INT8:
      return ExpectEqual<int8>(x, y);
    case DT_STRING:
      return ExpectEqual<tstring>(x, y);
    case DT_COMPLEX64:
      return ExpectEqual<complex64>(x, y, t);
    case DT_COMPLEX128:
      return ExpectEqual<complex128>(x, y, t);
    case DT_INT64:
      return ExpectEqual<int64_t>(x, y);
    case DT_UINT64:
      return ExpectEqual<uint64>(x, y);
    case DT_BOOL:
      return ExpectEqual<bool>(x, y);
    case DT_QINT8:
      return ExpectEqual<qint8>(x, y);
    case DT_QUINT8:
      return ExpectEqual<quint8>(x, y);
    case DT_QINT16:
      return ExpectEqual<qint16>(x, y);
    case DT_QUINT16:
      return ExpectEqual<quint16>(x, y);
    case DT_QINT32:
      return ExpectEqual<qint32>(x, y);
    case DT_BFLOAT16:
      return ExpectEqual<bfloat16>(x, y, t);
    case DT_HALF:
      return ExpectEqual<Eigen::half>(x, y, t);
    case DT_FLOAT8_E5M2:
      return ExpectEqual<float8_e5m2>(x, y, t);
    case DT_FLOAT8_E4M3FN:
      return ExpectEqual<float8_e4m3fn>(x, y, t);
    default:
      EXPECT_TRUE(false) << "Unsupported type : " << DataTypeString(x.dtype());
  }
}

void ExpectClose(const Tensor& x, const Tensor& y, double atol, double rtol) {
  ASSERT_TRUE(IsSameType(x, y));
  ASSERT_TRUE(IsSameShape(x, y));

  switch (x.dtype()) {
    case DT_HALF:
      return ExpectClose<Eigen::half>(x, y, atol, rtol);
    case DT_BFLOAT16:
      return ExpectClose<Eigen::bfloat16>(x, y, atol, rtol);
    case DT_FLOAT:
      return ExpectClose<float>(x, y, atol, rtol);
    case DT_DOUBLE:
      return ExpectClose<double>(x, y, atol, rtol);
    case DT_COMPLEX64:
      return ExpectClose<complex64>(x, y, atol, rtol);
    case DT_COMPLEX128:
      return ExpectClose<complex128>(x, y, atol, rtol);
    default:
      EXPECT_TRUE(false) << "Unsupported type : " << DataTypeString(x.dtype());
  }
}

::testing::AssertionResult internal_test::IsClose(Eigen::half x, Eigen::half y,
                                                  double atol, double rtol) {
  return test::IsClose(x, y, GetTolerance<Eigen::half>(atol),
                       GetTolerance<Eigen::half>(rtol));
}
::testing::AssertionResult internal_test::IsClose(float x, float y, double atol,
                                                  double rtol) {
  return test::IsClose(x, y, GetTolerance<float>(atol),
                       GetTolerance<float>(rtol));
}
::testing::AssertionResult internal_test::IsClose(double x, double y,
                                                  double atol, double rtol) {
  return test::IsClose(x, y, GetTolerance<double>(atol),
                       GetTolerance<double>(rtol));
}

}  // end namespace test
}  // end namespace tensorflow
