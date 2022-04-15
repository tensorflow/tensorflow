/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_DATASET_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/service/common.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace data {

// A resource which counts how many iterators have been created. This is used
// by the DataServiceDataset to coordinate jobs across multiple iterations.
class IterationCounter : public ResourceBase {
 public:
  IterationCounter() : counter_(0) {}

  std::string DebugString() const override {
    mutex_lock l(mu_);
    return absl::StrCat(counter_);
  }

  int64_t GetAndIncrement() {
    mutex_lock l(mu_);
    return ++counter_;
  }

 private:
  mutable mutex mu_;
  int64_t counter_ TF_GUARDED_BY(mu_) = 0;
};

// Creates a dataset for reading from the tf.data service.
class DataServiceDatasetOp : public DatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "DataService";
  static constexpr const char* const kDatasetId = "dataset_id";
  static constexpr const char* const kProcessingMode = "processing_mode";
  static constexpr const char* const kAddress = "address";
  static constexpr const char* const kProtocol = "protocol";
  static constexpr const char* const kDataTransferProtocol =
      "data_transfer_protocol";
  static constexpr const char* const kJobName = "job_name";
  static constexpr const char* const kConsumerIndex = "consumer_index";
  static constexpr const char* const kNumConsumers = "num_consumers";
  static constexpr const char* const kMaxOutstandingRequests =
      "max_outstanding_requests";
  static constexpr const char* const kTaskRefreshIntervalHintMs =
      "task_refresh_interval_hint_ms";
  static constexpr const char* const kTargetWorkers = "target_workers";
  static constexpr const char* const kIterationCounter = "iteration_counter";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kUncompress = "uncompress";
  static constexpr const char* const kUncompressFn = "uncompress_fn";

  // Note: If a new constant is declared here, it *must* be defined in
  // data_service_dataset_op.cc, otherwise it will not compile in debug mode.

  explicit DataServiceDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;
  int op_version_;
  int64_t task_refresh_interval_hint_ms_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  std::string data_transfer_protocol_;
  TargetWorkers target_workers_ = TARGET_WORKERS_AUTO;
  bool uncompress_;
  std::shared_ptr<FunctionMetadata> uncompress_fn_ = nullptr;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_DATASET_OP_H_
