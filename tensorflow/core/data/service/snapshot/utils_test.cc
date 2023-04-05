/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/snapshot/utils.h"

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

TEST(UtilsTest, EstimatedSizeBytes) {
  // int64 Tensor of size 1000.
  Tensor tensor(DT_INT64, TensorShape({10, 100}));
  std::vector<Tensor> Tensors{Tensor(DT_INT64, TensorShape({10, 100}))};
  EXPECT_GT(EstimatedSizeBytes(Tensors), 1000);
}

TEST(UtilsTest, EstimatedVariantSizeBytes) {
  // Variant Tensor of size 1000.
  std::unique_ptr<CompressedElement> compressed{
      protobuf::Arena::CreateMessage<CompressedElement>(nullptr)};
  compressed->set_data(std::string(1000, 'a'));
  Tensor tensor(DT_VARIANT, TensorShape({}));
  tensor.scalar<Variant>()() = *compressed;

  EXPECT_GT(EstimatedSizeBytes({tensor}), 1000);
}

TEST(UtilsTest, EstimatedMixedElementsSizeBytes) {
  // int64 Tensor of size 1000.
  Tensor int64_tensor(DT_INT64, TensorShape({10, 100}));

  // Variant Tensor of size 1000.
  std::unique_ptr<CompressedElement> compressed{
      protobuf::Arena::CreateMessage<CompressedElement>(nullptr)};
  compressed->set_data(std::string(1000, 'a'));
  Tensor variant_tensor(DT_VARIANT, TensorShape({}));
  variant_tensor.scalar<Variant>()() = *compressed;

  EXPECT_GT(EstimatedSizeBytes({int64_tensor, variant_tensor}), 2000);
}

TEST(UtilsTest, EmptyTensor) { EXPECT_GT(EstimatedSizeBytes({Tensor()}), 0); }

TEST(UtilsTest, StreamAssignmentChanged) {
  Status stream_assignment_changed =
      StreamAssignmentChanged("worker_address", /*stream_index=*/0);
  EXPECT_TRUE(IsStreamAssignmentChanged(stream_assignment_changed));
  EXPECT_FALSE(IsStreamAssignmentChanged(OkStatus()));
  EXPECT_FALSE(IsStreamAssignmentChanged(
      errors::FailedPrecondition("Not stream assignment changed.")));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
