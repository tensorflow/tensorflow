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
  template <typename IndexT, typename DataT>
  void TestR1() {
    // Slice at dimension start.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {0}, {5}, {0, 1, 2, 3, 4});
    // Slice in the middle.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {2}, {3}, {2, 3, 4});
    // Slice at dimension boundaries.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {5}, {3}, {5, 6, 7});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {6}, {4}, {6, 7, 0, 1});
    // Zero element slice.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {2}, {0}, {});
  }

  template <typename IndexT, typename DataT>
  void TestR2() {
    // Slice at dimension start.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {2, 2},
                         {{1, 2}, {4, 5}});
    // Slice in the middle.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {2, 1},
                         {{5}, {8}});
    // Slice at dimension boundaries.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {2, 1},
                         {{5}, {8}});
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {1, 1}, {3, 3},
                         {{5, 6, 4}, {8, 9, 7}, {2, 3, 1}});
    // Zero element slice: 2x0.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {2, 0},
                         {{}, {}});
    // Zero element slice: 0x2.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {0, 0}, {0, 2},
                         Array2D<DataT>(0, 2));
  }

  template <typename IndexT, typename DataT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // clang-format off

    // Slice at dimension start.
    RunR3<IndexT, DataT>(
      {{{1, 2}, {3, 4}, {5, 6}},
       {{7, 8}, {9, 10}, {11, 12}}},
      {0, 0, 0}, {2, 1, 2},
      {{{1, 2}}, {{7, 8}}});

    // Slice in the middle.
    RunR3<IndexT, DataT>(
      {{{1, 2}, {3, 4}, {5, 6}},
       {{7, 8}, {9, 10}, {11, 12}}},
      {0, 1, 1}, {2, 2, 1},
      {{{4}, {6}}, {{10}, {12}}});

    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR3<IndexT, DataT>(
      {{{1, 2}, {3, 4}, {5, 6}},
       {{7, 8}, {9, 10}, {11, 12}}},
      {0, 2, 1}, {2, 1, 2},
      {{{6, 5}}, {{12, 11}}});

    // clang-format on
  }

  template <typename IndexT, typename DataT>
  void RunR1(tensorflow::gtl::ArraySlice<DataT> input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64>& slice_sizes,
             tensorflow::gtl::ArraySlice<DataT> expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR1<DataT>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR1<DataT>(&builder, expected_values, {start_data.get()});
  }

  template <typename IndexT, typename DataT>
  void RunR2(const Array2D<DataT>& input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64>& slice_sizes,
             const Array2D<DataT>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR2FromArray2D<DataT>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR2<DataT>(&builder, expected_values, {start_data.get()});
  }

  template <typename IndexT, typename DataT>
  void RunR3(const Array3D<DataT>& input_values,
             const std::vector<IndexT> slice_starts,
             const std::vector<int64>& slice_sizes,
             const Array3D<DataT>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR3FromArray3D<DataT>(input_values);
    builder.DynamicSlice(input, starts, slice_sizes);
    // Run computation and compare against expected values.
    ComputeAndCompareR3<DataT>(&builder, expected_values, {start_data.get()});
  }
};

XLA_TEST_F(DynamicSliceTest, Int32R1) { TestR1<int32, int32>(); }

XLA_TEST_F(DynamicSliceTest, Int64R1) { TestR1<int64, float>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R1) { TestR1<uint64, double>(); }

XLA_TEST_F(DynamicSliceTest, Int32R2) { TestR2<int32, float>(); }

XLA_TEST_F(DynamicSliceTest, Int64R2) { TestR2<int64, double>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R2) { TestR2<uint64, int32>(); }

XLA_TEST_F(DynamicSliceTest, Int32R3) { TestR3<int32, int32>(); }

XLA_TEST_F(DynamicSliceTest, Int64R3) { TestR3<int64, float>(); }

XLA_TEST_F(DynamicSliceTest, UInt64R3) { TestR3<uint64, double>(); }

XLA_TEST_F(DynamicSliceTest, Int32R1Pred) {
  // Slice at dimension start.
  RunR1<int32, bool>({true, false, false, true, false, true, true, false}, {0},
                     {5}, {true, false, false, true, false});
  // Slice in the middle.
  RunR1<int32, bool>({true, false, false, true, false, true, true, false}, {2},
                     {3}, {false, true, false});
  // Slice at dimension boundaries.
  RunR1<int32, bool>({true, false, false, true, false, true, true, false}, {5},
                     {3}, {true, true, false});
  // Zero element slice.
  RunR1<int32, bool>({true, false, false, true, false, true, true, false}, {2},
                     {0}, {});
}

XLA_TEST_F(DynamicSliceTest, Int32R2Pred) {
  // Slice at dimension start.
  RunR2<int32, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {2, 2}, {{true, false}, {false, false}});
  // Slice in the middle.
  RunR2<int32, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {1, 1},
      {2, 1}, {{false}, {true}});
  // Slice at dimension boundaries.
  RunR2<int32, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {1, 1},
      {2, 1}, {{false}, {true}});
  // Zero element slice: 2x0.
  RunR2<int32, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {2, 0}, {{}, {}});
  // Zero element slice: 0x2.
  RunR2<int32, bool>(
      {{true, false, true}, {false, false, true}, {true, true, false}}, {0, 0},
      {0, 2}, Array2D<bool>(0, 2));
}

XLA_TEST_F(DynamicSliceTest, Int32R3Pred) {
  // R3 Shape: [2, 3, 2]
  // clang-format off

  // Slice at dimension start.
  RunR3<int32, bool>(
    {{{true, false}, {false, true}, {true, true}},
     {{false, true}, {true, false}, {false, false}}},
    {0, 0, 0}, {2, 1, 2},
    {{{true, false}}, {{false, true}}});

  // Slice in the middle.
  RunR3<int32, bool>(
    {{{true, false}, {false, true}, {true, true}},
     {{false, true}, {true, false}, {false, false}}},
    {0, 1, 1}, {2, 2, 1},
    {{{true}, {true}}, {{false}, {false}}});

  // clang-format on
}

class DynamicUpdateSliceTest : public ClientLibraryTestBase {
 protected:
  template <typename IndexT, typename DataT>
  void TestR1() {
    // Slice at dimension start.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {0},
                         {8, 9, 10, 3, 4, 5, 6, 7});
    // Slice in the middle.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {2},
                         {0, 1, 8, 9, 10, 5, 6, 7});
    // Slice at dimension boundaries.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {5},
                         {0, 1, 2, 3, 4, 8, 9, 10});
    // Zero-sized update.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {}, {2},
                         {0, 1, 2, 3, 4, 5, 6, 7});
  }

  template <typename IndexT, typename DataT>
  void TestR2() {
    // Slice at dimension start.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {0, 0},
                         {{10, 11, 3}, {4, 5, 6}, {7, 8, 9}});
    // Slice in the middle.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {1, 1},
                         {{1, 2, 3}, {4, 10, 11}, {7, 8, 9}});
    // Slice at dimension boundaries.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {2, 1},
                         {{1, 2, 3}, {4, 5, 6}, {7, 10, 11}});
    // Zero-sized update.
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{}}, {2, 1},
                         {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  }

  template <typename IndexT, typename DataT>
  void TestR3() {
    // R3 Shape: [2, 3, 2]
    // Slice at dimension start.
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}},
        {{{13, 14}, {15, 16}}, {{17, 18}, {19, 20}}}, {0, 0, 0},
        {{{13, 14}, {15, 16}, {5, 6}}, {{17, 18}, {19, 20}, {11, 12}}});
    // Slice in the middle.
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}, {{{13}, {15}}},
        {1, 1, 1}, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 13}, {11, 15}}});
  }

  template <typename IndexT, typename DataT>
  void TestWrap() {
    // Slice at dimension boundaries, but with sizes that cause indices to wrap.
    RunR1<IndexT, DataT>({0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10}, {6},
                         {10, 1, 2, 3, 4, 5, 8, 9});
    // R2 Shape: [3, 3]
    RunR2<IndexT, DataT>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, {{10, 11}}, {2, 2},
                         {{1, 2, 3}, {4, 5, 6}, {11, 8, 10}});
    // R3 Shape: [2, 3, 2]
    RunR3<IndexT, DataT>(
        {{{1, 2}, {3, 4}, {5, 6}}, {{7, 8}, {9, 10}, {11, 12}}}, {{{13}, {15}}},
        {1, 2, 1}, {{{1, 2}, {3, 4}, {5, 6}}, {{7, 15}, {9, 10}, {11, 13}}});
  }

  template <typename IndexT, typename DataT>
  void RunR1(tensorflow::gtl::ArraySlice<DataT> input_values,
             tensorflow::gtl::ArraySlice<DataT> update_values,
             const std::vector<IndexT> slice_starts,
             tensorflow::gtl::ArraySlice<DataT> expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR1<DataT>(input_values);
    auto update = builder.ConstantR1<DataT>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR1<DataT>(&builder, expected_values, {start_data.get()});
  }

  template <typename IndexT, typename DataT>
  void RunR2(const Array2D<DataT>& input_values,
             const Array2D<DataT>& update_values,
             const std::vector<IndexT> slice_starts,
             const Array2D<DataT>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR2FromArray2D<DataT>(input_values);
    auto update = builder.ConstantR2FromArray2D<DataT>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR2<DataT>(&builder, expected_values, {start_data.get()});
  }

  template <typename IndexT, typename DataT>
  void RunR3(const Array3D<DataT>& input_values,
             const Array3D<DataT>& update_values,
             const std::vector<IndexT> slice_starts,
             const Array3D<DataT>& expected_values) {
    ComputationBuilder builder(client_, TestName());
    // Initialize and transfer dynamic slice start indices parameter.
    ComputationDataHandle starts;
    std::unique_ptr<GlobalData> start_data = CreateR1Parameter<IndexT>(
        slice_starts, 0, "slice_starts", &builder, &starts);
    // Build dynamic slice computation.
    auto input = builder.ConstantR3FromArray3D<DataT>(input_values);
    auto update = builder.ConstantR3FromArray3D<DataT>(update_values);
    builder.DynamicUpdateSlice(input, update, starts);
    // Run computation and compare against expected values.
    ComputeAndCompareR3<DataT>(&builder, expected_values, {start_data.get()});
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
        Literal::CreateR3FromArray3D<NativeT>(values);
    LOG(INFO) << name << ":" << literal->ToString();
  }
};

XLA_TEST_F(DynamicUpdateSliceTest, Int32R1) { TestR1<int32, float>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R1) { TestR1<int64, float>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R1) { TestR1<uint64, double>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32R2) { TestR2<int32, float>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R2) { TestR2<int64, int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R2) { TestR2<uint64, int32>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32R3) { TestR3<int32, float>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64R3) { TestR3<int64, int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64R3) { TestR3<uint64, uint64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32Wrap) { TestWrap<int32, float>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int64Wrap) { TestWrap<int64, int64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, UInt64Wrap) { TestWrap<uint64, uint64>(); }

XLA_TEST_F(DynamicUpdateSliceTest, Int32R1Pred) {
  // Slice at dimension start.
  RunR1<int32, bool>({false, false, true, true, false, true, true, false},
                     {true, true, false}, {0},
                     {true, true, false, true, false, true, true, false});
  // Slice in the middle.
  RunR1<int32, bool>({false, false, true, true, false, true, true, false},
                     {false, true, true}, {2},
                     {false, false, false, true, true, true, true, false});
  // Slice at dimension boundaries.
  RunR1<int32, bool>({false, false, true, true, false, true, true, false},
                     {false, true, true}, {5},
                     {false, false, true, true, false, false, true, true});
  // Zero-sized update.
  RunR1<int32, bool>({false, false, true, true, false, true, true, false}, {},
                     {2}, {false, false, true, true, false, true, true, false});
}

XLA_TEST_F(DynamicUpdateSliceTest, Int32R2Pred) {
  // Slice at dimension start.
  RunR2<int32, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {0, 0},
      {{true, false, false}, {true, false, true}, {false, true, true}});
  // Slice in the middle.
  RunR2<int32, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {1, 1},
      {{false, true, false}, {true, true, false}, {false, true, true}});
  // Slice at dimension boundaries.
  RunR2<int32, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}},
      {{true, false}}, {2, 1},
      {{false, true, false}, {true, false, true}, {false, true, false}});
  // Zero-sized update.
  RunR2<int32, bool>(
      {{false, true, false}, {true, false, true}, {false, true, true}}, {{}},
      {2, 1}, {{false, true, false}, {true, false, true}, {false, true, true}});
}

XLA_TEST_F(DynamicUpdateSliceTest, Int32R3Pred) {
  // R3 Shape: [2, 3, 2]
  // Slice at dimension start.
  RunR3<int32, bool>(
      {{{true, false}, {false, true}, {true, true}},
       {{false, false}, {false, true}, {true, false}}},
      {{{false, true}, {true, false}}, {{true, true}, {false, true}}},
      {0, 0, 0},
      {{{false, true}, {true, false}, {true, true}},
       {{true, true}, {false, true}, {true, false}}});
  // Slice in the middle.
  RunR3<int32, bool>({{{true, false}, {false, true}, {true, true}},
                      {{false, false}, {false, true}, {true, false}}},
                     {{{false}, {true}}}, {1, 1, 1},
                     {{{true, false}, {false, true}, {true, true}},
                      {{false, false}, {false, false}, {true, true}}});
}

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

XLA_TEST_F(DynamicUpdateSliceTest, R3ContiguousMultipleWrapping) {
  // Multiple element, wrapping.
  std::vector<int32> operand_shape({4, 5, 2});
  RunR3Contiguous(operand_shape, /*index=*/3, /*size=*/2);
}

XLA_TEST_F(DynamicUpdateSliceTest, R3ContiguousTooLarge) {
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
  auto input_literal = Literal::CreateR4(
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
  auto buffer = client->backend()
                    .transfer_manager()
                    ->AllocateScopedShapedBuffer(
                        start_indices_shape, &allocator, /*device_ordinal=*/0)
                    .ConsumeValueOrDie();

  auto start_indices_literal = Literal::CreateR1<int32>({0, 1, 2, 3});
  ASSERT_IS_OK(transfer_manager->TransferLiteralToDevice(
      executors[device_ordinal], *start_indices_literal, *buffer));

  std::unique_ptr<LocalExecutable> executable =
      client
          ->Compile(computation, {&buffer->on_host_shape()},
                    ExecutableBuildOptions())
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
