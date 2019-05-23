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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "optimize_dataset";
constexpr char kOpName[] = "OptimizeDataset";

class OptimizeDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `OptimizeDataset` op kernel.
  Status CreateOptimizeDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      const std::vector<string>& optimization_configs,
      std::unique_ptr<OpKernel>* optimize_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, kOpName, {"input_dataset", "optimizations"},
        {{"output_types", output_types},
         {"output_shapes", output_shapes},
         {"optimization_configs", optimization_configs}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, optimize_dataset_op_kernel));
    return Status::OK();
  }

  // Create a new `OptimizeDataset` op kernel context.
  Status CreateOptimizeDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }

  // Create the input dataset as a variant tensor for OptimizeDataset.
  Status MakeInputDataset(const string& dataset_type, int64 count,
                          const DataTypeVector& output_types,
                          const std::vector<PartialTensorShape>& output_shapes,
                          Tensor* input_dataset) {
    GraphConstructorOptions graph_opts;
    graph_opts.allow_internal_ops = true;
    graph_opts.expect_device_spec = false;

    Tensor range_dataset_tensor;
    Tensor start = CreateTensor<int64>(TensorShape({}), {-3});
    Tensor stop = CreateTensor<int64>(TensorShape({}), {3});
    Tensor step = CreateTensor<int64>(TensorShape({}), {1});
    TF_RETURN_IF_ERROR(
        RunFunction(test::function::MakeRangeDataset(),
                    /*attrs*/
                    {{"output_types", DataTypeVector({DT_INT64})},
                     {"output_shapes",
                      std::vector<PartialTensorShape>{PartialTensorShape({})}}},
                    /*inputs*/ {start, stop, step}, graph_opts,
                    /*rets*/ {&range_dataset_tensor}));

    if (dataset_type == "TakeDataset") {
      Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
      TF_RETURN_IF_ERROR(RunFunction(
          test::function::MakeTakeDataset(),
          /*attrs*/
          {{"output_types", output_types}, {"output_shapes", output_shapes}},
          /*inputs*/ {range_dataset_tensor, count_tensor}, graph_opts,
          /*rets*/ {input_dataset}));
    } else if (dataset_type == "SkipDataset") {
      Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
      TF_RETURN_IF_ERROR(RunFunction(
          test::function::MakeSkipDataset(),
          /*attrs*/
          {{"output_types", output_types}, {"output_shapes", output_shapes}},
          /*inputs*/ {range_dataset_tensor, count_tensor}, graph_opts,
          /*rets*/ {input_dataset}));
    } else if (dataset_type == "PrefetchDataset") {
      Tensor buffer_size = CreateTensor<int64>(TensorShape({}), {count});
      TF_RETURN_IF_ERROR(RunFunction(
          test::function::MakePrefetchDataset(),
          /*attrs*/
          {{"output_types", output_types},
           {"output_shapes", output_shapes},
           {"slack_period", 0}},
          /*inputs*/ {range_dataset_tensor, buffer_size}, graph_opts,
          /*rets*/ {input_dataset}));
    } else if (dataset_type == "RepeatDataset") {
      Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
      TF_RETURN_IF_ERROR(RunFunction(
          test::function::MakeRepeatDataset(),
          /*attrs*/
          {{"output_types", output_types}, {"output_shapes", output_shapes}},
          /*inputs*/ {range_dataset_tensor, count_tensor}, graph_opts,
          /*rets*/ {input_dataset}));
    }

    return Status::OK();
  }
};

struct NoopEliminationTestCase {
  string dataset_type;
  int64 count;
  DataTypeVector output_types;
  std::vector<PartialTensorShape> output_shapes;
  std::vector<Tensor> expected_outputs;
};

template <typename T>
std::vector<Tensor> ConvertToTensorVec(std::vector<T> values) {
  std::vector<Tensor> tensors;
  tensors.reserve(values.size());
  for (auto& value : values) {
    tensors.emplace_back(
        DatasetOpsTestBase::CreateTensor<T>(TensorShape({}), {value}));
  }
  return tensors;
}

NoopEliminationTestCase NoopEliminationTestCase1() {
  return {/*dataset_type*/ "TakeDataset",
          /*count*/ -3,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase2() {
  return {/*dataset_type*/ "TakeDataset",
          /*count*/ -1,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase3() {
  return {/*dataset_type*/ "TakeDataset",
          /*count*/ 0,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/ {}};
}

NoopEliminationTestCase NoopEliminationTestCase4() {
  return {/*dataset_type*/ "TakeDataset",
          /*count*/ 3,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1})};
}

NoopEliminationTestCase NoopEliminationTestCase5() {
  return {/*dataset_type*/ "SkipDataset",
          /*count*/ -1,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/ {}};
}

NoopEliminationTestCase NoopEliminationTestCase6() {
  return {/*dataset_type*/ "SkipDataset",
          /*count*/ 0,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase7() {
  return {/*dataset_type*/ "SkipDataset",
          /*count*/ 3,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase8() {
  return {/*dataset_type*/ "PrefetchDataset",
          /*buffer_size*/ 0,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase9() {
  return {/*dataset_type*/ "PrefetchDataset",
          /*buffer_size*/ 1,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase10() {
  return {/*dataset_type*/ "RepeatDataset",
          /*count*/ 1,
          /*output_types*/ DataTypeVector({DT_INT64}),
          /*output_shapes*/
          std::vector<PartialTensorShape>{PartialTensorShape({})},
          /*expected_outputs*/
          ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2})};
}

NoopEliminationTestCase NoopEliminationTestCase11() {
  return {
      /*dataset_type*/ "RepeatDataset",
      /*count*/ 2,
      /*output_types*/ DataTypeVector({DT_INT64}),
      /*output_shapes*/
      std::vector<PartialTensorShape>{PartialTensorShape({})},
      /*expected_outputs*/
      ConvertToTensorVec<int64>({-3, -2, -1, 0, 1, 2, -3, -2, -1, 0, 1, 2})};
}

class ParameterizedOptimizeDatasetOpWithNoopEliminationTest
    : public OptimizeDatasetOpTest,
      public ::testing::WithParamInterface<NoopEliminationTestCase> {};

TEST_P(ParameterizedOptimizeDatasetOpWithNoopEliminationTest, NoopElimination) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  NoopEliminationTestCase test_case = GetParam();
  Tensor input_dataset_tensor;
  TF_ASSERT_OK(MakeInputDataset(test_case.dataset_type, test_case.count,
                                test_case.output_types, test_case.output_shapes,
                                &input_dataset_tensor));

  std::unique_ptr<OpKernel> optimize_dataset_kernel;
  TF_ASSERT_OK(CreateOptimizeDatasetOpKernel(
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({})},
      /*optimization_configs*/ {}, &optimize_dataset_kernel));
  Tensor optimizations =
      CreateTensor<string>(TensorShape({1}), {"noop_elimination"});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&input_dataset_tensor, &optimizations});
  std::unique_ptr<OpKernelContext> optimize_dataset_context;
  TF_ASSERT_OK(CreateOptimizeDatasetContext(
      optimize_dataset_kernel.get(), &inputs, &optimize_dataset_context));

  DatasetBase* optimize_dataset;
  TF_ASSERT_OK(CreateDataset(optimize_dataset_kernel.get(),
                             optimize_dataset_context.get(),
                             &optimize_dataset));
  core::ScopedUnref scoped_unref(optimize_dataset);

  std::unique_ptr<IteratorContext> iterator_context;
  TF_ASSERT_OK(
      CreateIteratorContext(optimize_dataset_context.get(), &iterator_context));
  std::unique_ptr<IteratorBase> iterator;
  TF_ASSERT_OK(optimize_dataset->MakeIterator(iterator_context.get(),
                                              "Iterator", &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_context.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_outputs,
                           /*compare_order*/ true));
}

INSTANTIATE_TEST_SUITE_P(
    OptimizeDatasetOpTest,
    ParameterizedOptimizeDatasetOpWithNoopEliminationTest,
    ::testing::ValuesIn(std::vector<NoopEliminationTestCase>(
        {NoopEliminationTestCase1(), NoopEliminationTestCase2(),
         NoopEliminationTestCase3(), NoopEliminationTestCase4(),
         NoopEliminationTestCase5(), NoopEliminationTestCase6(),
         NoopEliminationTestCase7(), NoopEliminationTestCase8(),
         NoopEliminationTestCase9(), NoopEliminationTestCase10(),
         NoopEliminationTestCase11()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
