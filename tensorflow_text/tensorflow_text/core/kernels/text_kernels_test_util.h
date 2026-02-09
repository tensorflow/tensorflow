// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// GMock matchers for testing text kernels:
//   TensorHasShapeAndValues<DTYPE>({dim1, ..., dimN}, {v1, v2, ..., vN});
//   VectorEq<DTYPE>({v1, v2, ..., vN});
//   MatrixEq<DTYPE>({{v1_1, ..., v1_M}, ..., {vN_1, ..., vN_M}});
//   TensorHasShape({dim1, ..., dimN});

#ifndef TENSORFLOW_TEXT_CORE_KERNELS_TEXT_KERNELS_TEST_UTIL_H_
#define TENSORFLOW_TEXT_CORE_KERNELS_TEXT_KERNELS_TEST_UTIL_H_

#include <gmock/gmock.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"

namespace tensorflow {
namespace text_kernels_test_util {

// GMock MatcherInterface for testing tensor equality.
class TensorEqMatcher : public ::testing::MatcherInterface<Tensor> {
 public:
  explicit TensorEqMatcher(const Tensor& expect) : expect_(expect) {}
  bool MatchAndExplain(Tensor actual,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* gmock_os) const override;
  void DescribeNegationTo(::std::ostream* gmock_os) const override;

 private:
  Tensor expect_;
};

// GMock MatcherInterface for testing tensor shapes.
class TensorHasShapeMatcher : public ::testing::MatcherInterface<Tensor> {
 public:
  explicit TensorHasShapeMatcher(const TensorShape& expect) : expect_(expect) {}
  bool MatchAndExplain(Tensor actual,
                       ::testing::MatchResultListener* listener) const override;
  void DescribeTo(::std::ostream* gmock_os) const override;
  void DescribeNegationTo(::std::ostream* gmock_os) const override;

 private:
  TensorShape expect_;
};

// Returns a gmock matcher that checks whether a given tensor has the specified
// dtype, values, and shape.  dtype is specified using the template parameter.
// values are specified as a flattened vector.
// Example:
//   EXPECT_THAT(*GetOutput(0),
//               TensorHasShapeAndValues<int64>({3, 2}, {1, 2, 3, 4, 5, 6});
template <typename DTYPE>
::testing::Matcher<Tensor> TensorHasShapeAndValues(
    const TensorShape& shape, const std::vector<DTYPE>& values) {
  Tensor expect = test::AsTensor<DTYPE>(values, shape);
  // MakeMatcher takes ownership of the TensorEqMatcher.
  return ::testing::MakeMatcher(new TensorEqMatcher(expect));
}

// Returns a gmock matcher that checks whether a given tensor is a 1-D tensor
// with the specified dtype and values.  dtype is specified using the template
// parameter.
// Example:
//   EXPECT_THAT(*GetOutput(0),
//               VectorEq<int64>({1, 2, 3, 4, 5, 6});
template <typename DTYPE>
::testing::Matcher<Tensor> VectorEq(const std::vector<DTYPE>& values) {
  int64 nvals = values.size();
  Tensor expect = test::AsTensor<DTYPE>(values, {nvals});
  // MakeMatcher takes ownership of the TensorEqMatcher.
  return ::testing::MakeMatcher(new TensorEqMatcher(expect));
}

// Returns a gmock matcher that checks whether a given tensor is a 2-D tensor
// with the specified dtype and values.  dtype is specified using the template
// parameter.  values are specified as a nested vector.  All rows of the values
// vector must have the same length.  The values vector may not be empty,
// since we can't infer the number of columns for an empty matrix; to test
// empty matrices, use the more general TensorHasShapeAndValues() instead.
// Example:
//   EXPECT_THAT(*GetOutput(0),
//               MatrixEq<int64>({{1, 2, 3}, {4, 5, 6}});
template <typename DTYPE>
::testing::Matcher<Tensor> MatrixEq(
    const std::vector<std::vector<DTYPE>>& values) {
  int64 nrows = values.size();
  CHECK_GT(nrows, 0)  // Crash OK
      << "Invalid use of MatrixEq: to test empty matrices, use "
      << "TensorHasShapeAndValues<dtype>{{0, ndims}, {}} instead.";
  int64 ncols = values[0].size();
  std::vector<DTYPE> flat;
  for (const auto& row : values) {
    CHECK_EQ(ncols, row.size())  // Crash OK
        << "Invalid use of MatrixEq: all rows must have equal length";
    flat.insert(flat.end(), row.begin(), row.end());
  }
  Tensor expect = test::AsTensor<DTYPE>(flat, TensorShape({nrows, ncols}));
  // MakeMatcher takes ownership of the TensorEqMatcher.
  return ::testing::MakeMatcher(new TensorEqMatcher(expect));
}

// Returns a gmock matcher that checks whether a given tensor has a specified
// shape.
// Example:
//   EXPECT_THAT(*GetOutput(0), TensorHasShape({2, 8});
::testing::Matcher<Tensor> TensorHasShape(const TensorShape& shape);

}  // namespace text_kernels_test_util
}  // namespace tensorflow

#endif  // TENSORFLOW_TEXT_CORE_KERNELS_TEXT_KERNELS_TEST_UTIL_H_
