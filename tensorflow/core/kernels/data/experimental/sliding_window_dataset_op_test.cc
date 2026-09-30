/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace data {
namespace experimental {
namespace {

constexpr char kNodeName[] = "sliding_window_dataset";

class SlidingWindowDatasetParams : public DatasetParams {
 public:
  template <typename T>
  SlidingWindowDatasetParams(T input_dataset_params, int64_t size,
                             int64_t shift, int64_t stride, bool drop_remainder,
                             DataTypeVector output_dtypes,
                             std::vector<PartialTensorShape> output_shapes,
                             std::string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        size_(size),
        shift_(shift),
        stride_(stride),
        drop_remainder_(drop_remainder) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return {CreateTensor<int64_t>(TensorShape({}), {size_}),
            CreateTensor<int64_t>(TensorShape({}), {shift_}),
            CreateTensor<int64_t>(TensorShape({}), {stride_})};
  }

  absl::Status GetInputNames(
      std::vector<std::string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back("input_dataset");
    input_names->emplace_back("window_size");
    input_names->emplace_back("window_shift");
    input_names->emplace_back("window_stride");
    return absl::OkStatus();
  }

  absl::Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("drop_remainder", drop_remainder_);
    return absl::OkStatus();
  }

  std::string dataset_type() const override { return "SlidingWindow"; }

 private:
  int64_t size_;
  int64_t shift_;
  int64_t stride_;
  bool drop_remainder_;
};

class SlidingWindowDatasetOpTest : public DatasetOpsTestBase {};

SlidingWindowDatasetParams SlidingWindowDatasetParamsWithOverflowingSizeAndStride() {
  const int64_t large_size = (static_cast<int64_t>(1) << 32) + 1;
  const int64_t large_stride = (static_cast<int64_t>(1) << 32);
  return SlidingWindowDatasetParams(RangeDatasetParams(0, 3, 1),
                                    /*size=*/large_size,
                                    /*shift=*/1,
                                    /*stride=*/large_stride,
                                    /*drop_remainder=*/true,
                                    /*output_dtypes=*/{DT_VARIANT},
                                    /*output_shapes=*/{PartialTensorShape({})},
                                    /*node_name=*/kNodeName);
}

SlidingWindowDatasetParams SlidingWindowDatasetParamsWithAddOverflowingSizeAndStride() {
  const int64_t max_stride = std::numeric_limits<int64_t>::max();
  return SlidingWindowDatasetParams(RangeDatasetParams(0, 3, 1),
                                    /*size=*/2,
                                    /*shift=*/1,
                                    /*stride=*/max_stride,
                                    /*drop_remainder=*/true,
                                    /*output_dtypes=*/{DT_VARIANT},
                                    /*output_shapes=*/{PartialTensorShape({})},
                                    /*node_name=*/kNodeName);
}

// Regression test: large window_size and window_stride whose target buffer
// size (window_size - 1) * window_stride + 1 overflows int64. Before the fix,
// the overflow silently wrapped to a small value and emitted undersized
// windows. After the fix, dataset initialization must return InvalidArgument.
TEST_F(SlidingWindowDatasetOpTest, OverflowingTargetBufferSize) {
  auto dataset_params = SlidingWindowDatasetParamsWithOverflowingSizeAndStride();
  EXPECT_THAT(
      Initialize(dataset_params),
      tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("overflow")));
}

// Regression test: addition overflow when (window_size - 1) * window_stride + 1
// overflows int64 on the addition step with size=2 and stride=int64_max.
TEST_F(SlidingWindowDatasetOpTest, AddOverflowingTargetBufferSize) {
  auto dataset_params = SlidingWindowDatasetParamsWithAddOverflowingSizeAndStride();
  EXPECT_THAT(
      Initialize(dataset_params),
      tsl::testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("overflow")));
}

}  // namespace
}  // namespace experimental
}  // namespace data
}  // namespace tensorflow
