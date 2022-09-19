/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_DATA_SERVICE_TEST_UTIL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_TEST_UTIL_H_

#include <functional>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace testing {

// Returns a test dataset representing
// tf.data.Dataset.range(range). Useful for testing dataset graph execution.
DatasetDef RangeDataset(int64_t range);

// Returns a test dataset representing
// tf.data.Dataset.range(range).map(lambda x: x*x).
DatasetDef RangeSquareDataset(int64_t range);

// Returns a test dataset representing
// tf.data.Dataset.range(range).shard(SHARD_HINT, SHARD_HINT).
DatasetDef RangeDatasetWithShardHint(int64_t range);

// Returns a test dataset representing
// tf.data.Dataset.range(100000000).repeat().
DatasetDef InfiniteDataset();

// Returns a test dataset representing
// tf.data.Dataset.from_tensor_slices(["filenames"]).interleave(
//     lambda filepath: tf.data.TextLineDataset(filepath),
//     cycle_length=10)
StatusOr<DatasetDef> InterleaveTextlineDataset(
    const std::vector<tstring>& filenames,
    const std::vector<tstring>& contents);

// Repeatedly calls `f()`, blocking until `f()` returns `false`.
//
// Returns an error if `f()` returns an error.
Status WaitWhile(std::function<StatusOr<bool>()> f);

// TODO(b/229726259): Make EqualsProto available in Googletest
// (Public feature request: https://github.com/google/googletest/issues/1761).
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
      : expected_(expected.ShortDebugString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p,
                       ::testing::MatchResultListener*) const {
    return p.ShortDebugString() == expected_;
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
}  // namespace testing
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_TEST_UTIL_H_
