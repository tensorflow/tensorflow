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

#include "tensorflow_text/core/kernels/text_kernels_test_util.h"

using ::testing::MakeMatcher;
using ::testing::Matcher;
using ::testing::MatchResultListener;

namespace tensorflow {
namespace text_kernels_test_util {

bool TensorEqMatcher::MatchAndExplain(
    Tensor actual, ::testing::MatchResultListener* listener) const {
  string expect_values = expect_.SummarizeValue(expect_.NumElements());
  string actual_values = actual.SummarizeValue(actual.NumElements());
  if (expect_.dtype() != actual.dtype() || expect_.shape() != actual.shape() ||
      expect_values != actual_values) {
    *listener << "\n          dtype=" << DataTypeString(actual.dtype());
    *listener << "\n          shape=" << actual.shape().DebugString();
    *listener << "\n         values=" << actual_values;
    return false;
  }
  return true;
}

void TensorEqMatcher::DescribeTo(::std::ostream* gmock_os) const {
  *gmock_os << "dtype=" << DataTypeString(expect_.dtype())
            << "\n          shape=" << expect_.shape().DebugString()
            << "\n         values="
            << expect_.SummarizeValue(expect_.NumElements());
}

void TensorEqMatcher::DescribeNegationTo(::std::ostream* gmock_os) const {
  *gmock_os << "is not equal to " << expect_.DebugString();
}

bool TensorHasShapeMatcher::MatchAndExplain(
    Tensor actual, ::testing::MatchResultListener* listener) const {
  if (expect_ != actual.shape()) {
    *listener << "\n          shape=" << actual.shape().DebugString();
    return false;
  }
  return true;
}

void TensorHasShapeMatcher::DescribeTo(::std::ostream* gmock_os) const {
  *gmock_os << "shape=" << expect_.DebugString();
}

void TensorHasShapeMatcher::DescribeNegationTo(::std::ostream* gmock_os) const {
  *gmock_os << "shape!=" << expect_.DebugString();
}

Matcher<Tensor> TensorHasShape(const TensorShape& shape) {
  // MakeMatcher takes ownership of the TensorHasShapeMatcher.
  return MakeMatcher(new TensorHasShapeMatcher(shape));
}

}  // namespace text_kernels_test_util
}  // namespace tensorflow
