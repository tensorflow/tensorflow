/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor_matcher.h"

#include <stdint.h>

#include <complex>
#include <ostream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace {

using tensorflow::Tensor;

template <typename T>
bool MatchAndExplainPointwise(absl::Span<const T> value,
                              absl::Span<const T> target,
                              ::testing::MatchResultListener* listener) {
  auto matcher = ::testing::MatcherCast<absl::Span<const T>>(
      ::testing::Pointwise(::testing::Eq(), target));
  return matcher.MatchAndExplain(value, listener);
}

class TensorEqMatcherImpl : public ::testing::MatcherInterface<const Tensor&> {
 public:
  explicit TensorEqMatcherImpl(const Tensor& target) : target_(target) {}

  void DescribeTo(::std::ostream* os) const override {
    *os << "data type is " << tensorflow::DataTypeString(target_.dtype())
        << ", and shape is " << target_.shape();
    switch (target_.dtype()) {
#define CASE_TYPE(T)                                       \
  case tensorflow::DataTypeToEnum<T>::value: {             \
    *os << ", and tensor data ";                           \
    absl::Span<const T> data(target_.unaligned_flat<T>()); \
    ::testing::MatcherCast<absl::Span<const T>>(           \
        ::testing::Pointwise(::testing::Eq(), data))       \
        .DescribeTo(os);                                   \
    break;                                                 \
  }
      TF_CALL_POD_STRING_TYPES(CASE_TYPE);
#undef CASE_TYPE
      default: {
        DLOG(FATAL) << "TensorEq matcher unsupported dtype: "
                    << tensorflow::DataTypeString(target_.dtype());
      }
    }
  }

  void DescribeNegationTo(::std::ostream* os) const override {
    *os << "data type is not " << tensorflow::DataTypeString(target_.dtype())
        << ", or shape is not " << target_.shape();
    switch (target_.dtype()) {
#define CASE_TYPE(T)                                       \
  case tensorflow::DataTypeToEnum<T>::value: {             \
    *os << ", or tensor data ";                            \
    absl::Span<const T> data(target_.unaligned_flat<T>()); \
    ::testing::MatcherCast<absl::Span<const T>>(           \
        ::testing::Pointwise(::testing::Eq(), data))       \
        .DescribeNegationTo(os);                           \
    break;                                                 \
  }
      TF_CALL_POD_STRING_TYPES(CASE_TYPE);
#undef CASE_TYPE
      default: {
        DLOG(FATAL) << "TensorEq matcher unsupported dtype: "
                    << tensorflow::DataTypeString(target_.dtype());
      }
    }
  }

  bool MatchAndExplain(
      const Tensor& value,
      ::testing::MatchResultListener* listener) const override {
    const bool dtype_compare = value.dtype() == target_.dtype();
    *listener << "whose data type " << tensorflow::DataTypeString(value.dtype())
              << (dtype_compare ? " matches " : " doesn't match ")
              << tensorflow::DataTypeString(target_.dtype());

    const bool shape_compare = value.shape() == target_.shape();
    *listener << ", whose shape " << value.shape()
              << (shape_compare ? " matches " : " doesn't match ")
              << target_.shape();

    if (!dtype_compare || !shape_compare) {
      return false;
    }

    // For POD-types, Tensor comparison can be done by comparing buffer returned
    // by tensor_data() functions. However, that does not give useful debug
    // information when match fails. Therefore we switch on data type.
    bool result;
    switch (target_.dtype()) {
#define CASE_TYPE(T)                                                       \
  case tensorflow::DataTypeToEnum<T>::value: {                             \
    result = MatchAndExplainPointwise<T>(                                  \
        value.unaligned_flat<T>(), target_.unaligned_flat<T>(), listener); \
    break;                                                                 \
  }
      TF_CALL_POD_STRING_TYPES(CASE_TYPE);
#undef CASE_TYPE
      default: {
        DLOG(FATAL) << "TensorEq matcher unsupported dtype: "
                    << tensorflow::DataTypeString(target_.dtype());
        result = false;
      }
    }

    return result;
  }

 private:
  const Tensor target_;
};

}  // namespace

TensorEq::operator ::testing::Matcher<const Tensor&>() const {
  return ::testing::MakeMatcher(new TensorEqMatcherImpl(target_));
}

}  // namespace test
}  // namespace tensorflow
