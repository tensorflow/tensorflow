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

  // Create a new `OptimizeDataset` op kernel context
  Status CreateOptimizeDatasetContext(
      OpKernel* const op_kernel,
      gtl::InlinedVector<TensorValue, 4>* const inputs,
      std::unique_ptr<OpKernelContext>* context) {
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*op_kernel, *inputs));
    TF_RETURN_IF_ERROR(CreateOpKernelContext(op_kernel, inputs, context));
    return Status::OK();
  }
};

TEST_F(OptimizeDatasetOpTest, FilterFusion) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(
      {test::function::IsZero(), test::function::IsZeroV2()}, cpu_num));

  Tensor range_dataset_tensor;
  Tensor start = CreateTensor<int64>(TensorShape({}), {-3});
  Tensor stop = CreateTensor<int64>(TensorShape({}), {3});
  Tensor step = CreateTensor<int64>(TensorShape({}), {1});

  GraphConstructorOptions graph_opts;
  graph_opts.allow_internal_ops = true;
  graph_opts.expect_device_spec = false;
  TF_EXPECT_OK(
      RunFunction(test::function::MakeRangeDataset(),
                  /*attrs*/
                  {{"output_types", DataTypeVector({DT_INT64})},
                   {"output_shapes",
                    std::vector<PartialTensorShape>{PartialTensorShape({})}}},
                  /*inputs*/ {start, stop, step}, graph_opts,
                  /*rets*/ {&range_dataset_tensor}));

  Tensor filter_dataset_tensor1;
  TF_EXPECT_OK(RunFunction(
      test::function::MakeFilterDataset(),
      /*attrs*/
      {// TODO(feihugis): make `IsZero` work here.
       {"predicate", FunctionDefHelper::FunctionRef("IsZeroV2", {})},
       {"Targuments", DataTypeVector({})},
       {"output_types", DataTypeVector({DT_INT64})},
       {"output_shapes",
        std::vector<PartialTensorShape>{PartialTensorShape({})}}},
      /*inputs*/ {range_dataset_tensor}, graph_opts,
      /*rets*/ {&filter_dataset_tensor1}));

  Tensor filter_dataset_tensor2;
  TF_EXPECT_OK(RunFunction(
      test::function::MakeFilterDataset(),
      /*attrs*/
      {{"predicate",
        FunctionDefHelper::FunctionRef("IsZero", {{"T", DT_INT64}})},
       {"Targuments", DataTypeVector({})},
       {"output_types", DataTypeVector({DT_INT64})},
       {"output_shapes",
        std::vector<PartialTensorShape>{PartialTensorShape({})}}},
      /*inputs*/ {filter_dataset_tensor1}, graph_opts,
      /*rets*/ {&filter_dataset_tensor2}));

  std::unique_ptr<OpKernel> optimize_dataset_kernel;
  TF_ASSERT_OK(CreateOptimizeDatasetOpKernel(
      /*output_dtypes*/ {DT_INT64},
      /*output_shapes*/ {PartialTensorShape({})},
      /*optimization_configs*/ {}, &optimize_dataset_kernel));
  Tensor optimizations =
      CreateTensor<string>(TensorShape({1}), {"filter_fusion"});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {&filter_dataset_tensor2, &optimizations});
  std::unique_ptr<OpKernelContext> optimize_dataset_context;
  TF_ASSERT_OK(CreateOptimizeDatasetContext(
      optimize_dataset_kernel.get(), &inputs, &optimize_dataset_context));

  DatasetBase* optimize_dataset;
  TF_ASSERT_OK(CreateDataset(optimize_dataset_kernel.get(),
                             optimize_dataset_context.get(),
                             &optimize_dataset));
  core::ScopedUnref scoped_unref(optimize_dataset);
  EXPECT_EQ(optimize_dataset->type_string(), "FilterDataset");

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
    out_tensors.insert(out_tensors.begin(), next.begin(), next.end());
  }

  std::vector<Tensor> expected_outputs = {
      CreateTensor<int64>(TensorShape({}), {0})};
  TF_EXPECT_OK(
      ExpectEqual(out_tensors, expected_outputs, /*compare_order*/ true));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
