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
#include "tensorflow/core/kernels/data/optimize_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/range_dataset_op.h"
#include "tensorflow/core/kernels/data/take_dataset_op.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "optimize_dataset";
constexpr char kNoopElimination[] = "noop_elimination";
constexpr char kIteratorPrefix[] = "Iterator";

class OptimizeDatasetOpTest : public DatasetOpsTestBase {
 protected:
  // Creates a new `OptimizeDataset` op kernel.
  Status CreateOptimizeDatasetOpKernel(
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      const std::vector<string>& optimization_configs,
      std::unique_ptr<OpKernel>* optimize_dataset_op_kernel) {
    NodeDef node_def = test::function::NDef(
        kNodeName, name_utils::OpName(OptimizeDatasetOp::kDatasetType),
        {OptimizeDatasetOp::kInputDataset, OptimizeDatasetOp::kOptimizations},
        {{OptimizeDatasetOp::kOutputTypes, output_types},
         {OptimizeDatasetOp::kOutputShapes, output_shapes},
         {OptimizeDatasetOp::kOptimizationConfigs, optimization_configs}});
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

  // Create a `RangeDataset` dataset as a variant tensor.
  Status MakeRangeDataset(const Tensor& start, const Tensor& stop,
                          const Tensor& step,
                          const DataTypeVector& output_types,
                          const std::vector<PartialTensorShape>& output_shapes,
                          Tensor* range_dataset) {
    GraphConstructorOptions graph_opts;
    graph_opts.allow_internal_ops = true;
    graph_opts.expect_device_spec = false;
    TF_RETURN_IF_ERROR(
        RunFunction(test::function::MakeRangeDataset(),
                    /*attrs*/
                    {{RangeDatasetOp::kOutputTypes, output_types},
                     {RangeDatasetOp::kOutputShapes, output_shapes}},
                    /*inputs*/ {start, stop, step}, graph_opts,
                    /*rets*/ {range_dataset}));
    return Status::OK();
  }

  // Create a `TakeDataset` dataset as a variant tensor.
  Status MakeTakeDataset(const Tensor& input_dataset, int64 count,
                         const DataTypeVector& output_types,
                         const std::vector<PartialTensorShape>& output_shapes,
                         Tensor* take_dataset) {
    GraphConstructorOptions graph_opts;
    graph_opts.allow_internal_ops = true;
    graph_opts.expect_device_spec = false;

    Tensor count_tensor = CreateTensor<int64>(TensorShape({}), {count});
    TF_RETURN_IF_ERROR(
        RunFunction(test::function::MakeTakeDataset(),
                    /*attrs*/
                    {{TakeDatasetOp::kOutputTypes, output_types},
                     {TakeDatasetOp::kOutputShapes, output_shapes}},
                    /*inputs*/ {input_dataset, count_tensor}, graph_opts,
                    /*rets*/ {take_dataset}));
    return Status::OK();
  }
};

TEST_F(OptimizeDatasetOpTest, NoopElimination) {
  int thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  Tensor range_dataset_tensor;
  DataTypeVector output_types = DataTypeVector({DT_INT64});
  std::vector<PartialTensorShape> output_shapes =
      std::vector<PartialTensorShape>{PartialTensorShape({})};
  Tensor start = CreateTensor<int64>(TensorShape({}), {-3});
  Tensor stop = CreateTensor<int64>(TensorShape({}), {3});
  Tensor step = CreateTensor<int64>(TensorShape({}), {1});
  TF_ASSERT_OK(MakeRangeDataset(start, stop, step, output_types, output_shapes,
                                &range_dataset_tensor));

  Tensor take_dataset_tensor;
  int count = -3;
  TF_ASSERT_OK(MakeTakeDataset(range_dataset_tensor, count, output_types,
                               output_shapes, &take_dataset_tensor));

  std::unique_ptr<OpKernel> optimize_dataset_kernel;
  TF_ASSERT_OK(CreateOptimizeDatasetOpKernel(output_types, output_shapes,
                                             /*optimization_configs*/ {},
                                             &optimize_dataset_kernel));
  Tensor optimizations =
      CreateTensor<string>(TensorShape({1}), {kNoopElimination});
  gtl::InlinedVector<TensorValue, 4> inputs(
      {TensorValue(&take_dataset_tensor), TensorValue(&optimizations)});
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
                                              kIteratorPrefix, &iterator));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator->GetNext(iterator_context.get(), &next, &end_of_sequence));
    out_tensors.insert(out_tensors.end(), next.begin(), next.end());
  }

  std::vector<Tensor> expected_outputs = {
      CreateTensor<int64>(TensorShape({}), {-3}),
      CreateTensor<int64>(TensorShape({}), {-2}),
      CreateTensor<int64>(TensorShape({}), {-1}),
      CreateTensor<int64>(TensorShape({}), {0}),
      CreateTensor<int64>(TensorShape({}), {1}),
      CreateTensor<int64>(TensorShape({}), {2})};
  TF_EXPECT_OK(ExpectEqual(out_tensors, expected_outputs,
                           /*compare_order*/ true));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
