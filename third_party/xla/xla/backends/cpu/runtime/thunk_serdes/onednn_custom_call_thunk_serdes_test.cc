/* Copyright 2026 The OpenXLA Authors.

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
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/onednn_config.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/casts.h"

namespace xla::cpu {
namespace {

// Serializes a `OneDnnOpThunk` into a `ThunkSequenceProto` and deserializes it
// back, returning the reconstructed thunk.
absl::StatusOr<std::unique_ptr<OneDnnOpThunk>> RoundTrip(
    std::unique_ptr<OneDnnOpThunk> thunk,
    const std::vector<BufferAllocation>& buffer_allocations) {
  ThunkSequence sequence;
  sequence.push_back(std::move(thunk));

  ThunkSequenceSerDesProtobuf serializer;
  ABSL_ASSIGN_OR_RETURN(ThunkSequenceProto proto, serializer.ToProto(sequence));

  ThunkSequenceSerDesProtobuf deserializer(/*hlo_module=*/nullptr,
                                           &buffer_allocations);
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ThunkSequence> deserialized,
                   deserializer.FromProto(proto));

  CHECK_EQ(deserialized->size(), 1);
  std::unique_ptr<Thunk> deserialized_thunk = std::move((*deserialized)[0]);
  CHECK_EQ(deserialized_thunk->kind(), Thunk::Kind::kCustomCall);
  return std::unique_ptr<OneDnnOpThunk>(
      tsl::down_cast<OneDnnOpThunk*>(deserialized_thunk.release()));
}

// Verifies that the target, the buffer slices/shapes and `is_tuple_result`
// survive a serialization round trip. `config` is checked by the caller because
// its type is target specific.
void ExpectRoundTripPreservesBuffers(absl::string_view target,
                                     const OneDnnOpThunk::OpBuffers& expected,
                                     const OneDnnOpThunk& actual_thunk) {
  const OneDnnOpThunk::OpBuffers& actual = actual_thunk.op_buffers();
  EXPECT_EQ(actual_thunk.target(), target);
  // `OneDnnOpThunk::Create` tags `op_name` with the target exactly once, so a
  // round trip must not accumulate the suffix.
  EXPECT_EQ(actual_thunk.info().op_name, target);
  EXPECT_EQ(actual.arguments_buffers, expected.arguments_buffers);
  EXPECT_EQ(actual.arguments_shapes, expected.arguments_shapes);
  EXPECT_EQ(actual.results_buffers, expected.results_buffers);
  EXPECT_EQ(actual.results_shapes, expected.results_shapes);
  EXPECT_EQ(actual.is_tuple_result, expected.is_tuple_result);
}

class OneDnnOpThunkSerdesTest : public ::testing::Test {
 protected:
  // Creates two argument buffers and one result buffer, and returns op buffers
  // referring to them. `is_tuple_result` is intentionally set to true to make
  // sure it is not silently dropped by the serdes.
  OneDnnOpThunk::OpBuffers CreateOpBuffers() {
    literals_.push_back(LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}}));
    literals_.push_back(LiteralUtil::CreateR2<float>({{5.0, 6.0}, {7.0, 8.0}}));
    literals_.push_back(LiteralUtil::CreateR2<float>({{0.0, 0.0}, {0.0, 0.0}}));

    OneDnnOpThunk::OpBuffers op_buffers;
    for (int64_t i = 0; i < literals_.size(); ++i) {
      buffer_allocations_.push_back(CreateBufferAllocation(i, literals_[i]));
    }
    for (int64_t i = 0; i < 2; ++i) {
      op_buffers.arguments_buffers.push_back(
          CreateBufferAllocationSlice(buffer_allocations_[i]));
      op_buffers.arguments_shapes.push_back(literals_[i].shape());
    }
    op_buffers.results_buffers.push_back(
        CreateBufferAllocationSlice(buffer_allocations_[2]));
    op_buffers.results_shapes.push_back(literals_[2].shape());
    op_buffers.is_tuple_result = true;
    return op_buffers;
  }

  std::vector<Literal> literals_;
  std::vector<BufferAllocation> buffer_allocations_;
};

TEST_F(OneDnnOpThunkSerdesTest, MatMul) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  OneDnnMatMulConfig config;
  config.set_transpose_a(true);
  config.set_transpose_b(false);

  ASSERT_OK_AND_ASSIGN(auto thunk,
                       OneDnnOpThunk::Create("__onednn$matmul", Thunk::Info(),
                                             op_buffers, config));
  ASSERT_OK_AND_ASSIGN(auto round_tripped,
                       RoundTrip(std::move(thunk), buffer_allocations_));

  ExpectRoundTripPreservesBuffers("__onednn$matmul", op_buffers,
                                  *round_tripped);
  ASSERT_TRUE(std::holds_alternative<OneDnnMatMulConfig>(
      round_tripped->onednn_op_config()));
  EXPECT_EQ(std::get<OneDnnMatMulConfig>(round_tripped->onednn_op_config())
                .SerializeAsString(),
            config.SerializeAsString());
}

TEST_F(OneDnnOpThunkSerdesTest, Convolution) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  OneDnnConvolutionConfig config;
  config.set_dims(4);
  config.set_feature_groups(1);

  ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$convolution", Thunk::Info(),
                                        op_buffers, config));
  ASSERT_OK_AND_ASSIGN(auto round_tripped,
                       RoundTrip(std::move(thunk), buffer_allocations_));

  ExpectRoundTripPreservesBuffers("__onednn$convolution", op_buffers,
                                  *round_tripped);
  ASSERT_TRUE(std::holds_alternative<OneDnnConvolutionConfig>(
      round_tripped->onednn_op_config()));
  EXPECT_EQ(std::get<OneDnnConvolutionConfig>(round_tripped->onednn_op_config())
                .SerializeAsString(),
            config.SerializeAsString());
}

TEST_F(OneDnnOpThunkSerdesTest, LayerNorm) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  OneDnnNormConfig config;
  config.set_rescale(OneDnnNormConfig::SCALE_AND_SHIFT);
  config.set_epsilon_typecast(1);

  ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$layernorm", Thunk::Info(),
                                        op_buffers, config));
  ASSERT_OK_AND_ASSIGN(auto round_tripped,
                       RoundTrip(std::move(thunk), buffer_allocations_));

  ExpectRoundTripPreservesBuffers("__onednn$layernorm", op_buffers,
                                  *round_tripped);
  ASSERT_TRUE(std::holds_alternative<OneDnnNormConfig>(
      round_tripped->onednn_op_config()));
  EXPECT_EQ(std::get<OneDnnNormConfig>(round_tripped->onednn_op_config())
                .SerializeAsString(),
            config.SerializeAsString());
}

TEST_F(OneDnnOpThunkSerdesTest, Softmax) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  OneDnnSoftmaxConfig config;
  config.set_softmax_axis(1);

  ASSERT_OK_AND_ASSIGN(auto thunk,
                       OneDnnOpThunk::Create("__onednn$softmax", Thunk::Info(),
                                             op_buffers, config));
  ASSERT_OK_AND_ASSIGN(auto round_tripped,
                       RoundTrip(std::move(thunk), buffer_allocations_));

  ExpectRoundTripPreservesBuffers("__onednn$softmax", op_buffers,
                                  *round_tripped);
  ASSERT_TRUE(std::holds_alternative<OneDnnSoftmaxConfig>(
      round_tripped->onednn_op_config()));
  EXPECT_EQ(std::get<OneDnnSoftmaxConfig>(round_tripped->onednn_op_config())
                .SerializeAsString(),
            config.SerializeAsString());
}

TEST_F(OneDnnOpThunkSerdesTest, PreservesThunkInfo) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  Thunk::Info info;
  info.op_name = "custom-call.1";
  info.module_name = "test_module";
  info.module_id = 42;

  ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$matmul", info, op_buffers,
                                        OneDnnMatMulConfig()));
  EXPECT_EQ(thunk->info().op_name, "custom-call.1__onednn$matmul");

  // Round trip twice to make sure the target suffix is not accumulated.
  ASSERT_OK_AND_ASSIGN(auto round_tripped,
                       RoundTrip(std::move(thunk), buffer_allocations_));
  ASSERT_OK_AND_ASSIGN(
      round_tripped, RoundTrip(std::move(round_tripped), buffer_allocations_));

  EXPECT_EQ(round_tripped->target(), "__onednn$matmul");
  EXPECT_EQ(round_tripped->info().op_name, "custom-call.1__onednn$matmul");
  EXPECT_EQ(round_tripped->info().module_name, "test_module");
  EXPECT_EQ(round_tripped->info().module_id, 42);
}

TEST_F(OneDnnOpThunkSerdesTest, UnsupportedTargetFailsToSerialize) {
  OneDnnOpThunk::OpBuffers op_buffers = CreateOpBuffers();

  ASSERT_OK_AND_ASSIGN(auto thunk,
                       OneDnnOpThunk::Create("__onednn$unknown", Thunk::Info(),
                                             op_buffers, OneDnnMatMulConfig()));

  ThunkSequence sequence;
  sequence.push_back(std::move(thunk));
  ThunkSequenceSerDesProtobuf serializer;
  EXPECT_FALSE(serializer.ToProto(sequence).ok());
}

}  // namespace
}  // namespace xla::cpu
