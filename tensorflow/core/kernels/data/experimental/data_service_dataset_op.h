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

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/resource_mgr.h"

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

  int64 GetAndIncrement() {
    mutex_lock l(mu_);
    return ++counter_;
  }

 private:
  mutable mutex mu_;
  int64 counter_ TF_GUARDED_BY(mu_) = 0;
};

// Creates a dataset for reading from the tf.data service.
class DataServiceDatasetOp : public DatasetOpKernel {
 public:
// Reminder: Avoid declare static constant variables in class, which will result in import error (Symbol not found in flat namespace) in macOS. 
// Define them in CC file instead.

  explicit DataServiceDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override;

 private:
  class Dataset;

  int64 task_refresh_interval_hint_ms_;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_DATA_SERVICE_DATASET_OP_H_
