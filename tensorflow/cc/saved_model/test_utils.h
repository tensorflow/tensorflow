/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_CC_SAVED_MODEL_TEST_UTILS_H_
#define TENSORFLOW_CC_SAVED_MODEL_TEST_UTILS_H_

#include <ostream>
#include <string>

#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow::saved_model {

// TODO(b/229726259) Switch to OSS version after it's available.
// Simple implementation of a proto matcher comparing string representations.
// Only works as ShapeProto's textual representation is deterministic.
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
      : expected_(expected.DebugString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    return p.DebugString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

}  // namespace tensorflow::saved_model

#endif  // TENSORFLOW_CC_SAVED_MODEL_TEST_UTILS_H_
