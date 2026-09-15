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
#include "tensorflow/core/kernels/data/iterator_ops.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::monitoring::testing::CellReader;
using ::tensorflow::monitoring::testing::Histogram;

class IteratorOpsTest : public DatasetOpsTestBase {
 public:
  absl::StatusOr<core::RefCountPtr<IteratorResource>> GetIteratorResource() {
    FunctionLibraryRuntime* flr = nullptr;
    std::unique_ptr<DeviceMgr> device_mgr;
    std::unique_ptr<FunctionLibraryDefinition> flib_def;
    std::unique_ptr<ProcessFunctionLibraryRuntime> plfr;
    TF_RETURN_IF_ERROR(dataset_ctx_->function_library()->Clone(
        &flib_def, &plfr, &flr, /*skip_flib_def=*/true));

    core::RefCountPtr<IteratorResource> iter_resource(
        new IteratorResource(dataset_ctx_->env(), dataset_->output_dtypes(),
                             dataset_->output_shapes(), std::move(device_mgr),
                             std::move(flib_def), std::move(plfr), flr));
    TF_RETURN_IF_ERROR(
        iter_resource->SetIteratorFromDataset(dataset_ctx_.get(), dataset_));
    return iter_resource;
  }

  absl::StatusOr<std::vector<std::vector<Tensor>>> GetIteratorOutput(
      IteratorResource& iterator) {
    std::vector<std::vector<Tensor>> output;
    for (bool end_of_sequence = false; !end_of_sequence;) {
      std::vector<Tensor> tensors;
      TF_RETURN_IF_ERROR(
          iterator.GetNext(dataset_ctx_.get(), &tensors, &end_of_sequence));
      if (end_of_sequence) {
        break;
      }
      output.push_back(std::move(tensors));
    }
    return output;
  }
};

TEST_F(IteratorOpsTest, CollectMetrics) {
  CellReader<Histogram> latency("/tensorflow/data/getnext_duration");
  CellReader<Histogram> iterator_gap("/tensorflow/data/iterator_gap");
  CellReader<int64_t> throughput("/tensorflow/data/bytes_fetched");
  CellReader<int64_t> iterator_lifetime("/tensorflow/data/iterator_lifetime");
  CellReader<int64_t> iterator_busy("/tensorflow/data/iterator_busy");
  EXPECT_FLOAT_EQ(latency.Delta().num(), 0.0);
  EXPECT_FLOAT_EQ(iterator_gap.Delta().num(), 0.0);
  EXPECT_EQ(throughput.Delta(), 0.0);
  EXPECT_EQ(iterator_lifetime.Delta(), 0.0);
  EXPECT_EQ(iterator_busy.Delta(), 0.0);

  RangeDatasetParams dataset_params = RangeDatasetParams(0, 10, 3);
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK_AND_ASSIGN(core::RefCountPtr<IteratorResource> iter_resource,
                          GetIteratorResource());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::vector<Tensor>> output,
                          GetIteratorOutput(*iter_resource));
  EXPECT_EQ(output.size(), 4);

  Histogram latency_histogram = latency.Delta();
  EXPECT_FLOAT_EQ(latency_histogram.num(), 5.0);
  EXPECT_GT(latency_histogram.sum(), 0.0);
  Histogram iterator_gap_histogram = iterator_gap.Delta();
  EXPECT_FLOAT_EQ(iterator_gap_histogram.num(), 5.0);
  EXPECT_GT(iterator_gap_histogram.sum(), 0.0);
  EXPECT_GT(throughput.Delta(), 0);
  EXPECT_GT(iterator_lifetime.Delta(), 0);
  EXPECT_GT(iterator_busy.Delta(), 0.0);
}

TEST_F(IteratorOpsTest, MultiDeviceIteratorGetNextFromShardFailsBeforeInit) {
  RangeDatasetParams dataset_params = RangeDatasetParams(0, 10, 3);
  TF_ASSERT_OK(InitializeRuntime(dataset_params));
  DataTypeVector output_types = {DT_INT64};
  std::vector<PartialTensorShape> output_shapes = {PartialTensorShape({})};

  Tensor multi_device_iterator;
  {
    // Creates the resource without running MultiDeviceIteratorInit. This
    // context must be destroyed before the next CreateOpKernelContext call,
    // which replaces the params it points to.
    NodeDef node_def = test::function::NDef(
        "multi_device_iterator", "MultiDeviceIterator", /*inputs=*/{},
        {{"devices",
          std::vector<std::string>{"/job:a/replica:0/task:0/device:CPU:0"}},
         {"shared_name", "uninitialized_multi_device_iterator"},
         {"container", ""},
         {"output_types", output_types},
         {"output_shapes", output_shapes}});
    std::unique_ptr<OpKernel> kernel;
    TF_ASSERT_OK(CreateOpKernel(node_def, &kernel));
    absl::InlinedVector<TensorValue, 4> inputs;
    std::unique_ptr<OpKernelContext> context;
    TF_ASSERT_OK(CreateOpKernelContext(kernel.get(), &inputs, &context));
    TF_ASSERT_OK(RunOpKernel(kernel.get(), context.get()));
    multi_device_iterator = *context->mutable_output(0);
  }

  Tensor shard_num(DT_INT32, TensorShape({}));
  shard_num.scalar<int32_t>()() = 0;
  Tensor incarnation_id(DT_INT64, TensorShape({}));
  incarnation_id.scalar<int64_t>()() = 0;

  NodeDef node_def = test::function::NDef(
      "get_next_from_shard", "MultiDeviceIteratorGetNextFromShard",
      {"multi_device_iterator", "shard_num", "incarnation_id"},
      {{"output_types", output_types}, {"output_shapes", output_shapes}});

  std::unique_ptr<OpKernel> kernel;
  TF_ASSERT_OK(CreateOpKernel(node_def, &kernel));

  absl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&multi_device_iterator));
  inputs.push_back(TensorValue(&shard_num));
  inputs.push_back(TensorValue(&incarnation_id));

  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(CreateOpKernelContext(kernel.get(), &inputs, &context));

  absl::Status status = RunOpKernel(kernel.get(), context.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(status.message(),
            "GetNextFromShard() failed because the MultiDeviceIterator has not "
            "been initialized. Ensure that you have run the initializer "
            "operation for this MultiDeviceIterator before getting the next "
            "element.");
}

TEST_F(IteratorOpsTest, DeleteMultiDeviceIteratorRejectsEmptyHandle) {
  RangeDatasetParams dataset_params = RangeDatasetParams(0, 10, 3);
  TF_ASSERT_OK(InitializeRuntime(dataset_params));

  Tensor empty_resource_handle(DT_RESOURCE, TensorShape({0}));
  Tensor variant_deleter(DT_VARIANT, TensorShape({}));

  NodeDef node_def = test::function::NDef(
      "delete_multi_device_iterator", "DeleteMultiDeviceIterator",
      {"multi_device_iterator", "deleter"}, {{"N", 0}});

  std::unique_ptr<OpKernel> kernel;
  TF_ASSERT_OK(CreateOpKernel(node_def, &kernel));

  absl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&empty_resource_handle));
  inputs.push_back(TensorValue(&variant_deleter));

  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(CreateOpKernelContext(kernel.get(), &inputs, &context));

  absl::Status status = RunOpKernel(kernel.get(), context.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Empty resource handle");
}

TEST_F(IteratorOpsTest, DeleteIteratorRejectsEmptyHandle) {
  RangeDatasetParams dataset_params = RangeDatasetParams(0, 10, 3);
  TF_ASSERT_OK(InitializeRuntime(dataset_params));

  Tensor empty_resource_handle(DT_RESOURCE, TensorShape({0}));
  Tensor variant_deleter(DT_VARIANT, TensorShape({}));

  NodeDef node_def = test::function::NDef("delete_iterator", "DeleteIterator",
                                          {"handle", "deleter"});

  std::unique_ptr<OpKernel> kernel;
  TF_ASSERT_OK(CreateOpKernel(node_def, &kernel));

  absl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&empty_resource_handle));
  inputs.push_back(TensorValue(&variant_deleter));

  std::unique_ptr<OpKernelContext> context;
  TF_ASSERT_OK(CreateOpKernelContext(kernel.get(), &inputs, &context));

  absl::Status status = RunOpKernel(kernel.get(), context.get());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(), "Empty resource handle");
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
