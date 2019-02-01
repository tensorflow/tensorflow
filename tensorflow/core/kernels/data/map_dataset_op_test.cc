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
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
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

constexpr char kNodeName[] = "map_dataset";
constexpr char kOpName[] = "MapDataset";

class MapDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new MapDataset op kernel. The `input_dataset` parameter should be
  // same with the node name of the input dataset for the method
  // `CreateMapDatasetContext()`. `T` specifies the output dtype of MapDataset.
  template <typename T>
  Status CreateMapDatasetOpKernel(const string& input_dataset,
                                  const string& func_name,
                                  std::unique_ptr<OpKernel>* map_kernel) {
    FunctionDefHelper::AttrValueWrapper func =
        FunctionDefHelper::FunctionRef(func_name, {{"T", DT_INT64}});

    map_node_def_ = test::function::NDef(
        kNodeName, kOpName, {input_dataset},
        {{"f", func},
         {"Targuments", {}},
         {"output_shapes", gtl::ArraySlice<TensorShape>{{}}},
         {"output_types",
          gtl::ArraySlice<DataType>{tensorflow::DataTypeToEnum<T>::value}},
         {"use_inter_op_parallelism", true},
         {"preserve_cardinality", false}});
    TF_CHECK_OK(CreateOpKernel(map_node_def_, map_kernel));
    return Status::OK();
  }

  // Creates a new MapDataset op kernel context.
  Status CreateMapDatasetContext(
      DatasetBase* const input_dataset, OpKernel* const map_kernel,
      std::unique_ptr<OpKernelContext>* map_context) {
    map_inputs_.clear();
    // Save the input dataset into a variant tensor as the input of MapDataset.
    Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(input_dataset, &dataset_tensor));
    Variant variant = dataset_tensor.scalar<Variant>()();
    TF_RETURN_IF_ERROR(AddDatasetInputFromArray<Variant>(
        &map_inputs_, map_kernel->input_types(), TensorShape({}), {variant}));
    input_dataset->Ref();
    TF_RETURN_IF_ERROR(
        CreateOpKernelContext(map_kernel, &map_inputs_, map_context));
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*map_kernel, map_inputs_));
    return Status::OK();
  }

 private:
  NodeDef map_node_def_;
  gtl::InlinedVector<TensorValue, 4> map_inputs_;
};

struct GetNextTestParams {
  explicit GetNextTestParams(int64 input_start, int64 input_end,
                             int64 input_step, string input_func_name,
                             std::vector<int64> input_expected_values,
                             std::vector<FunctionDef> input_func_lib)
      : start(input_start),
        end(input_end),
        step(input_step),
        func_name(std::move(input_func_name)),
        expected_values(std::move(input_expected_values)),
        func_lib(std::move(input_func_lib)) {}

  int64 start;
  int64 end;
  int64 step;
  string func_name;
  std::vector<int64> expected_values;
  std::vector<FunctionDef> func_lib;
};

struct DatasetGetNextTest : MapDatasetOpTest,
                            ::testing::WithParamInterface<GetNextTestParams> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  GetNextTestParams test_params = GetParam();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_params.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(test_params.start, test_params.end,
                                         test_params.step, "range",
                                         &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), test_params.func_name, &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(map_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
  }

  EXPECT_EQ(out_tensors.size(), test_params.expected_values.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    int64 actual_value = out_tensors[i].flat<int64>()(0);
    int64 expect_value = test_params.expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

INSTANTIATE_TEST_CASE_P(
    MapDatasetOpTest, DatasetGetNextTest,
    ::testing::Values(
        GetNextTestParams(
            0, 10, 3, "XTimesTwo", std::vector<int64>{0, 6, 12, 18},
            std::vector<FunctionDef>{test::function::XTimesTwo()}),
        GetNextTestParams(0, 10, 3, "XAddX", std::vector<int64>{0, 6, 12, 18},
                          std::vector<FunctionDef>{test::function::XAddX()}),
        GetNextTestParams(
            10, 0, -3, "XTimesFour", std::vector<int64>{40, 28, 16, 4},
            std::vector<FunctionDef>{test::function::XTimesTwo(),
                                     test::function::XTimesFour()})));

TEST_F(MapDatasetOpTest, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  EXPECT_EQ(map_dataset->name(), kOpName);
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(map_dataset->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(map_dataset->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset->output_shapes().size(); ++i) {
    map_dataset->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
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
    : MapDatasetOpTest,
      ::testing::WithParamInterface<CardinalityTestParams> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  CardinalityTestParams test_params = GetParam();
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(test_params.start, test_params.end,
                                         test_params.step, "range",
                                         &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  EXPECT_EQ(map_dataset->Cardinality(), test_params.expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(MapDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(CardinalityTestParams(0, 10, 1, 10),
                                          CardinalityTestParams(0, 10, 3, 4),
                                          CardinalityTestParams(10, 0, -3, 4)));

TEST_F(MapDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<SerializationContext> serialization_context;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(map_dataset->Save(serialization_context.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(map_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(map_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset->output_shapes().size(); ++i) {
    iterator->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(MapDatasetOpTest, IteratorOutputPrefix) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(
      CreateRangeDataset<int64>(start, end, step, "range", &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), func_def.signature().name(), &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(map_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  EXPECT_EQ(iterator->prefix(), "Iterator::Map");
}

struct RoundtripTestParams {
  explicit RoundtripTestParams(int64 input_start, int64 input_end,
                               int64 input_step, int input_breakpoint,
                               int64 input_expected_value,
                               string input_func_name,
                               std::vector<FunctionDef> input_func_lib)
      : start(input_start),
        end(input_end),
        step(input_step),
        breakpoint(input_breakpoint),
        expected_value(input_expected_value),
        func_name(std::move(input_func_name)),
        func_lib(std::move(input_func_lib)) {}

  int64 start;
  int64 end;
  int64 step;
  int breakpoint;
  int64 expected_value;
  string func_name;
  std::vector<FunctionDef> func_lib;
};

struct IteratorRoundtripTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<RoundtripTestParams> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  RoundtripTestParams test_params = GetParam();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(test_params.func_lib, cpu_num));

  DatasetBase* range_dataset;
  TF_ASSERT_OK(CreateRangeDataset<int64>(test_params.start, test_params.end,
                                         test_params.step, "range",
                                         &range_dataset));
  core::ScopedUnref scored_unref_range_dataset(range_dataset);

  std::unique_ptr<OpKernel> map_kernel;
  TF_ASSERT_OK(CreateMapDatasetOpKernel<int64>(
      range_dataset->name(), test_params.func_name, &map_kernel));
  std::unique_ptr<OpKernelContext> map_context;
  TF_ASSERT_OK(
      CreateMapDatasetContext(range_dataset, map_kernel.get(), &map_context));
  DatasetBase* map_dataset;
  TF_ASSERT_OK(
      CreateDataset(map_kernel.get(), map_context.get(), &map_dataset));
  core::ScopedUnref scored_unref_map_dataset(map_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(CreateIteratorContext(map_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(
      map_dataset->MakeIterator(iterator_context.get(), "Iterator", &iterator));

  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  for (int i = 0; i < test_params.breakpoint; i++) {
    TF_EXPECT_OK(iterator->GetNext(iterator_context.get(), &out_tensors,
                                   &end_of_sequence));
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
  EXPECT_EQ(out_tensors.back().flat<int64>()(0), test_params.expected_value);
}

INSTANTIATE_TEST_CASE_P(
    MapDatasetOpTest, IteratorRoundtripTest,
    ::testing::Values(RoundtripTestParams(0, 10, 2, 0, 0, "XTimesTwo",
                                          std::vector<FunctionDef>{
                                              test::function::XTimesTwo()}),
                      RoundtripTestParams(0, 10, 2, 4, 16, "XAddX",
                                          std::vector<FunctionDef>{
                                              test::function::XAddX()}),
                      RoundtripTestParams(0, 10, 2, 6, 32, "XTimesFour",
                                          std::vector<FunctionDef>{
                                              test::function::XTimesTwo(),
                                              test::function::XTimesFour()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
