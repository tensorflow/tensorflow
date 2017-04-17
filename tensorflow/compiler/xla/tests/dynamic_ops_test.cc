/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/legacy_flags/cpu_compiler_flags.h"
#include "tensorflow/compiler/xla/reference_util.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace {

class DynamicSliceTest : public ClientLibraryTestBase {
 protected:
  template <typename IndexT>
  void TestR1() {
    // Slice at dimension start.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, {0}, {5},
                  {0.0, 1.0, 2.0, 3.0, 4.0});
    // Slice in the middle.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, {2}, {3},
                  {2.0, 3.0, 4.0});
    // Slice at dimension boundaries.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, {5}, {3},
                  {5.0, 6.0, 7.0});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, {6}, {4},
                  {6.0, 7.0, 0.0, 1.0});
  }

  template <typename IndexT>
  void TestR2() {
    // Slice at dimension start.
    RunR2<IndexT>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
                  {0, 0}, {2, 2}, {{1.0f, 2.0f}, {4.0f, 5.0f}});
    // Slice in the middle.
    RunR2<IndexT>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
                  {1, 1}, {2, 1}, {{5.0f}, {8.0f}});
    // Slice at dimension boundaries.
    RunR2<IndexT>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
                  {1, 1}, {2, 1}, {{5.0f}, {8.0f}});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR2<IndexT>({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
                  {1, 1}, {3, 3},
                  {{5.0f, 6.0f, 4.0f}, {8.0f, 9.0f, 7.0f}, {2.0f, 3.0f, 1.0f}});
  }

  template <typename IndexT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // clang-format off

    // Slice at dimension start.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
        {0, 0, 0}, {2, 1, 2},
      {{{1.0f, 2.0f}}, {{7.0f, 8.0f}}});

    // Slice in the middle.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
        {0, 1, 1}, {2, 2, 1},
      {{{4.0f}, {6.0f}}, {{10.0f}, {12.0f}}});

    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
        {0, 2, 1}, {2, 2, 1},
      {{{6.0f}, {2.0f}}, {{12.0f}, {8.0f}}});

    // clang-format on
  }

  template <typename IndexT>
  void RunR1(const std::vector<float>& input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64> slice_sizes,
             const std::vector<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR1<float>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR1<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }

  template <typename IndexT>
  void RunR2(const Array2D<float>& input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64> slice_sizes,
             const Array2D<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR2FromArray2D<float>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR2<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }

  template <typename IndexT>
  void RunR3(const Array3D<float>& input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64> slice_sizes,
             const Array3D<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR3FromArray3D<float>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR3<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }
};

XLA_TEST_F(DynamicSliceTest, Int32R1) { TestR1<int32>(); }

XLA_TEST_F(DynamicSliceTest, Int64R1) { TestR1<int64>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R1) { TestR1<uint64>(); }

XLA_TEST_F(DynamicSliceTest, Int32R2) { TestR2<int32>(); }

XLA_TEST_F(DynamicSliceTest, Int64R2) { TestR2<int64>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R2) { TestR2<uint64>(); }

XLA_TEST_F(DynamicSliceTest, Int32R3) { TestR3<int32>(); }

XLA_TEST_F(DynamicSliceTest, Int64R3) { TestR3<int64>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R3) { TestR3<uint64>(); }

class DynamicUpdateSliceTest : public ClientLibraryTestBase {
 protected:
  template <typename IndexT>
  void TestR1() {
    // clang-format off
    // Slice at dimension start.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                  {8.0, 9.0, 10.0}, {0},
                  {8.0, 9.0, 10.0, 3.0, 4.0, 5.0, 6.0, 7.0});
    // Slice in the middle.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                  {8.0, 9.0, 10.0}, {2},
                  {0.0, 1.0, 8.0, 9.0, 10.0, 5.0, 6.0, 7.0});
    // Slice at dimension boundaries.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                  {8.0, 9.0, 10.0}, {5},
                  {0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0, 10.0});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR1<IndexT>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0},
                  {8.0, 9.0, 10.0}, {6},
                  {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0});
    // clang-format on
  }

  template <typename IndexT>
  void TestR2() {
    // clang-format off
    // Slice at dimension start.
    RunR2<IndexT>(
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
        {{10.0f, 11.0f}}, {0, 0},
        {{10.0f, 11.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}});
    // Slice in the middle.
    RunR2<IndexT>(
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
        {{10.0f, 11.0f}}, {1, 1},
        {{1.0f, 2.0f, 3.0f}, {4.0f, 10.0f, 11.0f}, {7.0f, 8.0f, 9.0f}});
    // Slice at dimension boundaries.
    RunR2<IndexT>(
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
        {{10.0f, 11.0f}}, {2, 1},
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 10.0f, 11.0f}});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR2<IndexT>(
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}},
        {{10.0f, 11.0f}}, {2, 2},
        {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 10.0f}});
    // clang-format on
  }

  template <typename IndexT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // clang-format off
    // Slice at dimension start.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
      {{{13.0f, 14.0f}, {15.0f, 16.0f}},
       {{17.0f, 18.0f}, {19.0f, 20.0f}}},
        {0, 0, 0},
      {{{13.0f, 14.0f}, {15.0f, 16.0f}, {5.0f, 6.0f}},
       {{17.0f, 18.0f}, {19.0f, 20.0f}, {11.0f, 12.0f}}});
    // Slice in the middle.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
      {{{13.0f}, {15.0f}}},
        {1, 1, 1},
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        {{7.0f, 8.0f}, {9.0f, 13.0f}, {11.0f, 15.0f}}});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR3<IndexT>(
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
       {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}}},
      {{{13.0f}, {15.0f}}},
        {1, 2, 1},
      {{{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}},
        {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 13.0f}}});
    // clang-format on
  }

  template <typename IndexT>
  void RunR1(const std::vector<float>& input_values,
             const std::vector<float>& update_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR1<float>(input_values);
    auto update = builder.ConstantR1<float>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR1<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }

  template <typename IndexT>
  void RunR2(const Array2D<float>& input_values,
             const Array2D<float>& update_values,
             const std::vector<IndexT> slice_starts,
             const Array2D<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR2FromArray2D<float>(input_values);
    auto update = builder.ConstantR2FromArray2D<float>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR2<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }

  template <typename IndexT>
  void RunR3(const Array3D<float>& input_values,
             const Array3D<float>& update_values,
             const std::vector<IndexT> slice_starts,
             const Array3D<float>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR3FromArray3D<float>(input_values);
    auto update = builder.ConstantR3FromArray3D<float>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR3<float>(&builder, expected_values, {start_data.get()},
                               ErrorSpec(0.000001));
  }

  void RunR3Contiguous(std::vector<int32> operand_shape, int32 index,
                       int32 size) {
    const int32 kSeq = operand_shape[0];
    const int32 kBatch = operand_shape[1];
    const int32 kDim = operand_shape[2];
    Array3D<float> input_values(kSeq, kBatch, kDim);
    Array3D<float> update_values(size, kBatch, kDim);
    Array3D<float> expected_values(kSeq, kBatch, kDim);

    input_values.FillIota(0);
    float val = 1000;
    update_values.FillIota(val);

    // TODO(b/34128753) Expected values may vary depending on backend when
    // the update wraps. According to documentation, the results are technically
    // implementation specific where the update is out of bounds, and hence
    // we don't really know what to pass into ComputeAndCompareR3.
    expected_values.FillIota(0);
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < kBatch; j++) {
        for (int k = 0; k < kDim; k++) {
          expected_values((index + i) % kSeq, j, k) = val++;
        }
      }
    }
    if (VLOG_IS_ON(1)) {
      DumpArray<float>("input", input_values);
      DumpArray<float>("update", update_values);
      DumpArray<float>("expected", expected_values);
    }

    // Build dynamic slice computation.
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer input parameter.
    ComputationDataHandle input;
    std::unique_ptr<GlobalData> input_data = CreateR3Parameter<float>(
        input_values, 0, "input_values", &builder, &input);
    // Initialize and transfer update parameter.
    ComputationDataHandle update;
    std::unique_ptr<GlobalData> update_data = CreateR3Parameter<float>(
        update_values, 1, "update_values", &builder, &update);
    auto starts = builder.ConstantR1<int32>({index, 0, 0});
    builder.DynamicUpdateSlice(input, update, starts);

    // Run computation and compare against expected values.
    ComputeAndCompareR3<float>(&builder, expected_values,
                               {input_data.get(), update_data.get()},
                               ErrorSpec(0.000001));
  }

  template <typename NativeT>
  void DumpArray(const string& name, const Array3D<NativeT> values) {
    std::unique_ptr<Literal> literal =
        LiteralUtil::CreateR3FromArray3D<NativeT>(values);
    LOG(INFO) << name << ":" << LiteralUtil::ToString(*literal);
  }
};

XLA_TEST_F(DynamicUpdateSliceTest, Int32R1) { TestR1<int32>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R1) { TestR1<int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R1) { TestR1<uint64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32R2) { TestR2<int32>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R2) { TestR2<int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R2) { TestR2<uint64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32R3) { TestR3<int32>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R3) { TestR3<int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R3) { TestR3<uint64>(); }

// Tests for simple R3 case where the update is contiguous (i.e. the minor
// two dimensions are not sliced).
XLA_TEST_F(DynamicUpdateSliceTest, R3ContiguousSingleElement) {
  // Single element, no wrap.
  std::vector<int32> operand_shape({4, 5, 2});
  RunR3Contiguous(operand_shape, /*index=*/1, /*size=*/1);
}

XLA_TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleElements) {
  // Multiple element, no wrap.
  std::vector<int32> operand_shape({4, 5, 2});
  RunR3Contiguous(operand_shape, /*index=*/1, /*size=*/2);
}

// TODO(b/34128753) CPU and GPU failed on 2016-01-06.  Appears not to handle
// wrapping as expected.
XLA_TEST_F(DynamicUpdateSliceTest,
           DISABLED_ON_CPU(DISABLED_ON_GPU(R3ContiguousMultipleWrapping))) {
  // Multiple element, wrapping.
  std::vector<int32> operand_shape({4, 5, 2});
  RunR3Contiguous(operand_shape, /*index=*/3, /*size=*/2);
}

// TODO(b/34128753) CPU and GPU failed on 2016-01-06.  Appears not to handle
// wrapping as expected.
XLA_TEST_F(DynamicUpdateSliceTest,
           DISABLED_ON_CPU(DISABLED_ON_GPU(R3ContiguousTooLarge))) {
  // Multiple element, update size larger than operand.
  std::vector<int32> operand_shape({4, 5, 2});
  RunR3Contiguous(operand_shape, /*index=*/5, /*size=*/2);
}

XLA_TEST_F(DynamicUpdateSliceTest, R3ContiguousUnaligned) {
  std::vector<int32> operand_shape({3, 123, 247});
  RunR3Contiguous(operand_shape, /*index=*/1, /*size=*/1);
}

// TODO(b/34134076) Disabled on GPU 2016-01-06 due to out-of-memory error.
XLA_TEST_F(DynamicUpdateSliceTest, DISABLED_ON_GPU(R3ContiguousLarger)) {
  std::vector<int32> operand_shape({32, 128, 1024});
  RunR3Contiguous(operand_shape, /*index=*/7, /*size=*/1);
}

void BM_DynamicSlice(int num_iters) {
  tensorflow::testing::StopTiming();

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().ValueOrDie();
  auto executors = PlatformUtil::GetStreamExecutors(platform).ValueOrDie();
  StreamExecutorMemoryAllocator allocator(platform, executors);
  LocalClient* client =
      ClientLibrary::GetOrCreateLocalClient(platform).ValueOrDie();
  auto* transfer_manager =
      TransferManager::GetForPlatform(platform).ValueOrDie();
  int device_ordinal = client->default_device_ordinal();

  ComputationBuilder builder(client, "DynamicSlice");

  // Create input as a constant: shape [1, 2, 3, 4]
  auto input_literal = LiteralUtil::CreateR4(
      {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
        {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}});
  auto input = builder.ConstantLiteral(*input_literal);

  // Create dynamic slice start indices as a parameter: shape [4]
  auto start_indices_shape = ShapeUtil::MakeShape(S32, {4});
  auto start_indices =
      builder.Parameter(0, start_indices_shape, "start_indices");
  // Add DynamicSlice op to the computatation.
  builder.DynamicSlice(input, start_indices, {1, 1, 1, 1});
  auto computation = builder.Build().ConsumeValueOrDie();

  // Initialize and transfer parameter buffer.
  auto buffer = ScopedShapedBuffer::MakeScopedShapedBuffer(start_indices_shape,
                                                           &allocator, 0)
                    .ConsumeValueOrDie();

  auto start_indices_literal = LiteralUtil::CreateR1<int32>({0, 1, 2, 3});
  ASSERT_IS_OK(transfer_manager->TransferLiteralToDevice(
      executors[device_ordinal], *start_indices_literal,
      buffer->mutable_buffer({})));

  std::unique_ptr<LocalExecutable> executable =
      client->Compile(computation, {&buffer->shape()}, ExecutableBuildOptions())
          .ConsumeValueOrDie();

  // Run some warm-up executions.
  ExecutableRunOptions options;
  options.set_allocator(&allocator);
  const int kWarmups = 2;
  for (int i = 0; i < kWarmups; ++i) {
    auto result = executable->Run({buffer.get()}, options);
    ASSERT_TRUE(result.ok());
  }

  // Run benchmark.
  tensorflow::testing::StartTiming();
  for (int i = 0; i < num_iters; ++i) {
    auto result = executable->Run({buffer.get()}, options);
    ASSERT_TRUE(result.ok());
  }
}
BENCHMARK(BM_DynamicSlice);

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::vector<tensorflow::Flag> flag_list;
  xla::legacy_flags::AppendCpuCompilerFlags(&flag_list);
  xla::string usage = tensorflow::Flags::Usage(argv[0], flag_list);
  const bool parse_result = tensorflow::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    LOG(ERROR) << "\n" << usage;
    return 2;
  }
  testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n" << usage;
    return 2;
  }
  return RUN_ALL_TESTS();
}
