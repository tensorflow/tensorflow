/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public DatasetOpsTestBase {
 public:
  static const char* kNodeName;
  static const char* kOpName;

 protected:
  // Creates a new RangeDataset op kernel context.
  Status CreateRangeDatasetContext(
      int64 start, int64 end, int64 step, OpKernel* const range_kernel,
      std::unique_ptr<OpKernelContext>* range_context) {
    inputs_.clear();
    TF_RETURN_IF_ERROR(AddDatasetInputFromArray<int64>(
        &inputs_, range_kernel->input_types(), TensorShape({}), {start}));
    TF_RETURN_IF_ERROR(AddDatasetInputFromArray<int64>(
        &inputs_, range_kernel->input_types(), TensorShape({}), {end}));
    TF_RETURN_IF_ERROR(AddDatasetInputFromArray<int64>(
        &inputs_, range_kernel->input_types(), TensorShape({}), {step}));

    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(range_kernel, &inputs_, range_context));
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*range_kernel, inputs_));
    return Status::OK();
  }

 private:
  gtl::InlinedVector<TensorValue, 4> inputs_;
};

const char* RangeDatasetOpTest::kNodeName = "range_dataset";
const char* RangeDatasetOpTest::kOpName = "RangeDataset";

struct GetNextTestParams {
  explicit GetNextTestParams(int64 input_start, int64 input_end,
                             int64 input_step)
      : start(input_start), end(input_end), step(input_step) {}

  int64 start;
  int64 end;
  int64 step;
};

struct DatasetGetNextTest : RangeDatasetOpTest,
                            ::testing::WithParamInterface<GetNextTestParams> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  GetNextTestParams params = GetParam();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(params.start, params.end, params.step,
                                         range_kernel.get(), &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(range_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
  }
  std::vector<int> expected_values;
  for (int i = params.start; (params.end - i) * params.step > 0;
       i = i + params.step) {
    expected_values.reserve(1);
    expected_values.emplace_back(i);
  }
  EXPECT_EQ(out_tensors.size(), expected_values.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    int64 actual_value = out_tensors[i].flat<int64>()(0);
    int64 expect_value = expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetGetNextTest,
                        ::testing::Values(GetNextTestParams(0, 10, 1),
                                          GetNextTestParams(0, 10, 3),
                                          GetNextTestParams(10, 0, -1),
                                          GetNextTestParams(10, 0, -3)));

TEST_F(RangeDatasetOpTest, DatasetName) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  EXPECT_EQ(range_dataset->name(), kOpName);
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(range_dataset->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(range_dataset->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < range_dataset->output_shapes().size(); ++i) {
    range_dataset->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

struct CardinalityTestParams {
  explicit CardinalityTestParams(int64 input_start, int64 input_end,
                                 int64 input_step,
                                 int input_expected_cardinality)
      : start(input_start),
        end(input_end),
        step(input_step),
        expected_cardinality(input_expected_cardinality) {}

  int64 start;
  int64 end;
  int64 step;
  int expected_cardinality;
};

struct DatasetCardinalityTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<CardinalityTestParams> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  CardinalityTestParams params = GetParam();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(params.start, params.end, params.step,
                                         range_kernel.get(), &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  EXPECT_EQ(range_dataset->Cardinality(), params.expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(CardinalityTestParams(0, 10, 1, 10),
                                          CardinalityTestParams(0, 10, 3, 4),
                                          CardinalityTestParams(10, 0, -3, 4)));

TEST_F(RangeDatasetOpTest, DatasetSave) {
  int64 thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));

  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(range_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(range_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(range_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < range_dataset->output_shapes().size(); ++i) {
    iterator->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(start, end, step, range_kernel.get(),
                                         &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(range_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Range");
}

struct RoundtripTestParams {
  explicit RoundtripTestParams(int64 input_start, int64 input_end,
                               int64 input_step, int input_breakpoint)
      : start(input_start),
        end(input_end),
        step(input_step),
        breakpoint(input_breakpoint) {}

  int64 start;
  int64 end;
  int64 step;
  int breakpoint;
};

struct IteratorRoundtripTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<RoundtripTestParams> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  RoundtripTestParams params = GetParam();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  std::unique_ptr<OpKernel> range_kernel;
  TF_ASSERT_OK(CreateRangeDatasetOpKernel<int64>("range", &range_kernel));
  std::unique_ptr<OpKernelContext> range_context;
  TF_ASSERT_OK(CreateRangeDatasetContext(params.start, params.end, params.step,
                                         range_kernel.get(), &range_context));
  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateDataset(range_kernel.get(), range_context.get(), &range_dataset));
  core::ScopedUnref scored_unref(range_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(range_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(range_dataset->MakeIterator(iterator_context.get(), "Iterator",
                                           &iterator));

  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  int64 cur_val = params.start - params.step;
  for (int i = 0; i < params.breakpoint; i++) {
    if (!end_of_sequence) {
      TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                     &end_of_sequence));
      cur_val = ((params.end - cur_val - params.step) * params.step > 0)
                    ? cur_val + params.step
                    : cur_val;
    }
  }

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(iterator->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
  VariantTensorDataReader reader(&data);
  TF_ASSERT_OK(iterator->Restore(iterator_context.get(), &reader));
  TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                 &end_of_sequence));
  int64 expect_next = ((params.end - cur_val - params.step) * params.step > 0)
                        ? cur_val + params.step
                        : cur_val;
  EXPECT_EQ(out_tensors.back().flat<int64>()(0), expect_next);
}

INSTANTIATE_TEST_CASE_P(
    RangeDatasetOpTest, IteratorRoundtripTest,
    ::testing::Values(
        RoundtripTestParams(0, 10, 2, 0),    // unused_iterator
        RoundtripTestParams(0, 10, 2, 4),    // fully_used_iterator_increase
        RoundtripTestParams(10, 0, -2, 4),   // fully_used_iterator_decrease
        RoundtripTestParams(0, 10, 2, 6)));  // exhausted_iterator

}  // namespace
}  // namespace data
}  // namespace tensorflow
