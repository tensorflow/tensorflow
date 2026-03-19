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
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
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

}  // namespace
}  // namespace data
}  // namespace tensorflow
