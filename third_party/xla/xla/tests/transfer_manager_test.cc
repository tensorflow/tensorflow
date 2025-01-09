/* Copyright 2017 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/stream_pool.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/local_client_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/test_benchmark.h"

namespace xla {
namespace {

class TransferManagerTest : public LocalClientTestBase {
 protected:
  TransferManagerTest()
      : shape_size_fn_([this](const Shape& shape) {
          return transfer_manager_->GetByteSizeRequirement(shape);
        }) {
    stream_ptr_ = local_client_->mutable_backend()
                      ->BorrowStream(stream_executor_)
                      .value();
    stream_ = stream_ptr_.get();
  }

  ~TransferManagerTest() override = default;

  ScopedShapedBuffer AllocateDeviceBuffer(const Shape& shape) {
    return transfer_manager_
        ->AllocateScopedShapedBuffer(
            shape, GetOrCreateAllocator(local_client_->platform()),
            /*device_ordinal=*/0)
        .value();
  }

 protected:
  StreamPool::Ptr stream_ptr_;
  se::Stream* stream_;

 private:
  std::function<int64_t(const Shape&)> shape_size_fn_;
};

XLA_TEST_F(TransferManagerTest, TransferR0U32) {
  Literal literal = LiteralUtil::CreateR0<uint32_t>(42);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR0Equal<uint32_t>(42, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1F32) {
  Literal literal =
      LiteralUtil::CreateR1<float>({1.25f, 2.5f, -17.0f, -20.125f});
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>({1.25f, 2.5f, -17.0f, -20.125f},
                                        result);
}

XLA_TEST_F(TransferManagerTest, TransferR1F32AwkwardSizes) {
  // Test transferring R1s from 0 to kMaxR1Size. The goal is to find bugs
  // related to "awkwardly" sized R1s.
  constexpr int kMaxR1Size = (1 << 11);
  for (int i = 0; i < kMaxR1Size; ++i) {
    std::vector<float> inputs(i);
    std::iota(inputs.begin(), inputs.end(), 0);
    Literal literal = LiteralUtil::CreateR1<float>(inputs);
    const Shape& shape = literal.shape();
    auto device_buffer = AllocateDeviceBuffer(shape);

    // Round trip literal through device.
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                            device_buffer));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal result,
        transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

    LiteralTestUtil::ExpectR1Equal<float>(inputs, result);
  }
}

XLA_TEST_F(TransferManagerTest, TransferR1LargeF32) {
  std::vector<float> test_vector(1024 * 1024);
  std::iota(test_vector.begin(), test_vector.end(), 0);
  Literal literal = LiteralUtil::CreateR1<float>(test_vector);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR1Equal<float>(test_vector, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1LargeUnalignedF32) {
  std::vector<float> test_vector(1025);
  std::iota(test_vector.begin(), test_vector.end(), 0);
  Shape shape = ShapeUtil::MakeShape(F32, {1024});
  BorrowingLiteral literal(reinterpret_cast<const char*>(&test_vector[1]),
                           shape);
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  std::vector<float> expected_output(1024);
  std::iota(expected_output.begin(), expected_output.end(), 1);
  LiteralTestUtil::ExpectR1Equal<float>(expected_output, result);
}

XLA_TEST_F(TransferManagerTest, TransferR1U8) {
  const char* test_string = "0123456789abcdef";
  Literal literal = LiteralUtil::CreateR1U8(test_string);
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_EQ(result.GetR1U8AsString(), test_string);
}

XLA_TEST_F(TransferManagerTest, TransferR2F32) {
  Literal literal =
      LiteralUtil::CreateR2<float>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
  const Shape& shape = literal.shape();
  auto device_buffer = AllocateDeviceBuffer(shape);

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, result);
}

XLA_TEST_F(TransferManagerTest,
           TransferR2F32AndChangeLayoutTransferringToDevice) {
  Literal literal = LiteralUtil::CreateR2WithLayout<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, LayoutUtil::MakeLayout({0, 1}));
  const Shape ondevice_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {1, 0});
  auto device_buffer = AllocateDeviceBuffer(ondevice_shape);

  // Round trip literal through device. Set the on-device layout to something
  // different than the literal layout.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_FALSE(
      LayoutUtil::Equal(result.shape().layout(), literal.shape().layout()));
  LiteralTestUtil::ExpectR2Equal<float>(
      {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, result);
}

XLA_TEST_F(TransferManagerTest, TransferTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
       LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferEmptyTuple) {
  Literal literal = LiteralUtil::MakeTuple({});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferNestedTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-10.0f, 123.0f})});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferComplexValue) {
  Literal literal = LiteralUtil::CreateR1<complex64>(
      {complex64(1.0f, 2.0f), complex64(42.0f, -123.4f)});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferComplexValueInTuple) {
  Literal literal = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR1<complex64>(
           {complex64(1.0f, 2.0f), complex64(42.0f, -123.4f)}),
       LiteralUtil::CreateR1<int32_t>({1, 2, 3, 4, 5, 6}),
       LiteralUtil::CreateR0<complex64>(complex64(0.3f, -0.4f))});
  auto device_buffer = AllocateDeviceBuffer(literal.shape());

  // Round trip literal through device.
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

XLA_TEST_F(TransferManagerTest, TransferTokenFromDevice) {
  // "Copy" a token from the device. The token has no physical representation
  // so no copying is actually performed, but it shouldn't fail.
  // TODO(b/110532604): Add transferring the token to device when this is
  // supported.
  auto device_buffer = AllocateDeviceBuffer(ShapeUtil::MakeTokenShape());
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateToken(), result));
}

XLA_TEST_F(TransferManagerTest, OVERSIZE_ON_GRM(MultiStreamRoundTripSoak)) {
  const int64_t kIterationCount = 5000;
  Literal literal1 = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(123.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{1.0f, 2.0f}, {4.0f, 5.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -10.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-10.0f, 123.0f})});
  Literal literal2 = LiteralUtil::MakeTupleFromSlices(
      {LiteralUtil::CreateR0<float>(456.0f),
       LiteralUtil::MakeTupleFromSlices(
           {LiteralUtil::CreateR2<float>({{5.0f, 7.0f}, {9.0f, 4.0f}}),
            LiteralUtil::CreateR1<float>({44.0f, -11.0f, 3333333.3f})}),
       LiteralUtil::CreateR1<float>({-98.0f, 153.0f})});

  auto device_buffer1 = AllocateDeviceBuffer(literal1.shape());
  auto device_buffer2 = AllocateDeviceBuffer(literal2.shape());

  auto stream1 = stream_;
  auto stream2 = stream_->GetOrCreateSubStream().value();

  Literal result1, result2;

  // Round trip literals through device in multiple streams asynchronously.
  for (int i = 0; i < kIterationCount; ++i) {
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream1, literal1,
                                                            device_buffer1));
    ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream2, literal2,
                                                            device_buffer2));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal this_result1,
        transfer_manager_->TransferLiteralFromDevice(stream1, device_buffer1));
    TF_ASSERT_OK_AND_ASSIGN(
        Literal this_result2,
        transfer_manager_->TransferLiteralFromDevice(stream2, device_buffer2));
    result1 = std::move(this_result1);
    result2 = std::move(this_result2);
  }

  EXPECT_TRUE(LiteralTestUtil::Equal(literal1, result1));
  EXPECT_TRUE(LiteralTestUtil::Equal(literal2, result2));
}

// TODO(b/223222672): TPUs transfer literals using a different codepath.
XLA_TEST_F(TransferManagerTest, DISABLED_ON_TPU(TransferDynamicShape)) {
  TF_ASSERT_OK_AND_ASSIGN(
      Shape s, ParseShape("(s64[], s32[<=1048576,3], f32[<=1048576,48])"));

  Literal literal(s);
  literal.SetDynamicSize(/*dim_index=*/0, /*shape_index=*/{1},
                         /*size=*/1048574);
  literal.SetDynamicSize(/*dim_index=*/0, /*shape_index=*/{2},
                         /*size=*/1048575);
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{0})
                   .Populate<int64_t>(
                       [](absl::Span<const int64_t> indices) { return 42; }));
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{1})
                   .Populate<int32_t>([](absl::Span<const int64_t> indices) {
                     return indices[0] + indices[1];
                   }));
  ASSERT_IS_OK(MutableBorrowingLiteral(&literal, /*view_root=*/{2})
                   .Populate<float>([](absl::Span<const int64_t> indices) {
                     return indices[0] + indices[1];
                   }));

  // Round trip `literal` through device.
  ScopedShapedBuffer device_buffer = AllocateDeviceBuffer(literal.shape());
  ASSERT_IS_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                          device_buffer));
  TF_ASSERT_OK_AND_ASSIGN(
      Literal result,
      transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));

  // LiteralTestUtil::Equal doesn't compare dynamic shapes, so we need to check
  // them ourselves.
  EXPECT_EQ(literal.GetDynamicSize(/*dim_index=*/0, /*shape_index=*/{1}),
            result.GetDynamicSize(0, {1}));
  EXPECT_EQ(literal.GetDynamicSize(/*dim_index=*/0, /*shape_index=*/{2}),
            result.GetDynamicSize(0, {2}));
  EXPECT_TRUE(LiteralTestUtil::Equal(literal, result));
}

class TransferDeviceToHostBenchmark : public TransferManagerTest {
 public:
  using TransferManagerTest::TransferManagerTest;
  ~TransferDeviceToHostBenchmark() override {}

  void Run(::testing::benchmark::State& state, int num_tuple_elements,
           int array_size) {
    SetUp();

    std::vector<Literal> tuple_elements;
    for (int i = 0; i < num_tuple_elements; ++i) {
      tuple_elements.push_back(
          LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, array_size, array_size));
    }
    Literal literal = LiteralUtil::MakeTupleOwned(std::move(tuple_elements));
    auto device_buffer = AllocateDeviceBuffer(literal.shape());
    TF_CHECK_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                           device_buffer));
    for (auto s : state) {
      TF_ASSERT_OK_AND_ASSIGN(
          Literal result,
          transfer_manager_->TransferLiteralFromDevice(stream_, device_buffer));
    }
    TearDown();
  }

  void TestBody() override {}
};

class TransferHostToDeviceBenchmark : public TransferManagerTest {
 public:
  using TransferManagerTest::TransferManagerTest;
  ~TransferHostToDeviceBenchmark() override {}

  void Run(::testing::benchmark::State& state, int num_tuple_elements,
           int array_size) {
    SetUp();

    std::vector<Literal> tuple_elements;
    for (int i = 0; i < num_tuple_elements; ++i) {
      tuple_elements.push_back(
          LiteralUtil::CreateR2F32Linspace(0.0f, 1.0f, array_size, array_size));
    }
    Literal literal = LiteralUtil::MakeTupleOwned(std::move(tuple_elements));
    auto device_buffer = AllocateDeviceBuffer(literal.shape());

    for (auto s : state) {
      TF_CHECK_OK(transfer_manager_->TransferLiteralToDevice(stream_, literal,
                                                             device_buffer));
    }
    TearDown();
  }

  void TestBody() override {}
};

void BM_TransferDeviceToHost(::testing::benchmark::State& state) {
  const int num_tuple_elements = state.range(0);
  const int array_size = state.range(1);

  TransferDeviceToHostBenchmark bm;
  bm.Run(state, num_tuple_elements, array_size);
}

void BM_TransferHostToDevice(::testing::benchmark::State& state) {
  const int num_tuple_elements = state.range(0);
  const int array_size = state.range(1);

  TransferHostToDeviceBenchmark bm;
  bm.Run(state, num_tuple_elements, array_size);
}

BENCHMARK(BM_TransferHostToDevice)
    ->ArgPair(1, 256)
    ->ArgPair(1, 257)
    ->ArgPair(100, 256)
    ->ArgPair(100, 257);

BENCHMARK(BM_TransferDeviceToHost)
    ->ArgPair(1, 256)
    ->ArgPair(1, 257)
    ->ArgPair(100, 256)
    ->ArgPair(100, 257);

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  tsl::testing::RunBenchmarks();
  return RUN_ALL_TESTS();
}

}  // namespace
}  // namespace xla
