/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/flat_map_utils.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace data {

FlatMapRandomAccessHandler::FlatMapRandomAccessHandler(
    OpKernelContext* const ctx, const DatasetBase* input_dataset,
    CapturedFunction& captured_map_func)
    : input_dataset_(input_dataset),
      captured_map_func_(captured_map_func),
      unbounded_thread_pool_(ctx->env(),
                             "tf_data_flat_map_random_access_handler") {
  absl::Status status =
      ctx->function_library()->Clone(&flib_def_, &pflr_, &flr_, true);
  if (!status.ok()) {
    cumulative_cardinalities_ = std::move(status);
    return;
  }
  function_handle_cache_ =
      std::make_unique<FunctionHandleCache>(pflr_->GetFLR("/device:CPU:0"));
  IteratorContext::Params params(ctx);
  params.cancellation_manager = &cancellation_manager_;
  params.env = ctx->env();
  params.flr = flr_;
  params.function_handle_cache = function_handle_cache_.get();
  params.resource_mgr = &resource_mgr_;
  params.thread_factory = unbounded_thread_pool_.get_thread_factory();
  params.thread_pool = &unbounded_thread_pool_;
  ctx_ = std::make_unique<IteratorContext>(std::move(params));
}

FlatMapRandomAccessHandler::~FlatMapRandomAccessHandler() {
  for (DatasetBase* dataset : input_datasets_) {
    dataset->Unref();
  }
  input_datasets_.clear();
}

absl::StatusOr<int64_t> FlatMapRandomAccessHandler::Cardinality() {
  TF_RETURN_IF_ERROR(cumulative_cardinalities_.status());
  if (cumulative_cardinalities_->empty()) {
    cumulative_cardinalities_ = ComputeCardinalities();
  }
  TF_RETURN_IF_ERROR(cumulative_cardinalities_.status());
  return cumulative_cardinalities_->back();
}

absl::StatusOr<int64_t> FlatMapRandomAccessHandler::CumulativeCardinality(
    size_t index) {
  TF_RETURN_IF_ERROR(cumulative_cardinalities_.status());
  if (index >= cumulative_cardinalities_->size()) {
    return absl::OutOfRangeError(absl::StrCat(
        "Dataset index exceeds the number of input datasets. Got index: ",
        index, ", number of input datasets: ",
        cumulative_cardinalities_->size(), "."));
  }
  return (*cumulative_cardinalities_)[index];
}

absl::StatusOr<std::vector<int64_t>>
FlatMapRandomAccessHandler::ComputeCardinalities() {
  if (input_datasets_.empty()) {
    TF_ASSIGN_OR_RETURN(input_datasets_, MakeInputDatasets());
  }

  std::vector<int64_t> cumulative_cardinalities;
  cumulative_cardinalities.reserve(input_datasets_.size());
  for (size_t i = 0; i < input_datasets_.size(); ++i) {
    int64_t input_cardinality = input_datasets_[i]->Cardinality();
    if (input_cardinality == kInfiniteCardinality ||
        input_cardinality == kUnknownCardinality) {
      cumulative_cardinalities.push_back(input_cardinality);
      return cumulative_cardinalities;
    }
    int64_t cumulative_cardinality = input_cardinality;
    if (i > 0) {
      cumulative_cardinality += cumulative_cardinalities.back();
    }
    cumulative_cardinalities.push_back(cumulative_cardinality);
  }
  if (cumulative_cardinalities.empty()) {
    cumulative_cardinalities.push_back(0);
  }
  return cumulative_cardinalities;
}

absl::StatusOr<int64_t> FlatMapRandomAccessHandler::GetDatasetIndex(
    size_t element_position) {
  TF_ASSIGN_OR_RETURN(int64_t cardinality, Cardinality());
  if (cardinality < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to globally shuffle flat map dataset. Global shuffling "
        "requires finite cardinality. Got ",
        cardinality, "."));
  }
  if (element_position >= cardinality) {
    return absl::OutOfRangeError(absl::StrCat(
        "Element index exceeds the flat map dataset cardinality. Got index: ",
        element_position, ", cardinality: ", cardinality, "."));
  }
  return std::upper_bound(cumulative_cardinalities_->begin(),
                          cumulative_cardinalities_->end(), element_position) -
         cumulative_cardinalities_->begin();
}

absl::StatusOr<std::vector<std::unique_ptr<IteratorBase>>>
FlatMapRandomAccessHandler::MakeInputIterators(
    IteratorContext* ctx, const DatasetBaseIterator* parent,
    const std::string& prefix) {
  if (input_datasets_.empty()) {
    TF_ASSIGN_OR_RETURN(input_datasets_, MakeInputDatasets());
  }

  std::vector<std::unique_ptr<IteratorBase>> result;
  if (input_datasets_.empty()) {
    return result;
  }

  result.resize(input_datasets_.size());
  for (size_t i = 0; i < input_datasets_.size(); ++i) {
    TF_RETURN_IF_ERROR(input_datasets_[i]->MakeIterator(
        ctx, parent, absl::StrCat(prefix, "[", i, "]"), &result[i]));
  }
  return result;
}

absl::StatusOr<std::deque<DatasetBase*>>
FlatMapRandomAccessHandler::MakeInputDatasets() const {
  std::unique_ptr<IteratorBase> iterator;
  TF_RETURN_IF_ERROR(input_dataset_->MakeIterator(
      ctx_.get(), /*parent=*/nullptr, "Iterator", &iterator));

  std::unique_ptr<InstantiatedCapturedFunction> map_func;
  TF_RETURN_IF_ERROR(captured_map_func_.Instantiate(ctx_.get(), &map_func));

  absl::Mutex mu;
  std::deque<DatasetBase*> input_datasets;
  absl::Status status;  // Guarded by `mu`.
  std::vector<std::unique_ptr<tsl::Thread>> threads;
  while (true) {
    std::vector<Tensor> input_tensors;
    bool end_of_sequence = false;
    TF_RETURN_IF_ERROR(
        iterator->GetNext(ctx_.get(), &input_tensors, &end_of_sequence));
    if (end_of_sequence) {
      break;
    }

    input_datasets.push_back(nullptr);
    DatasetBase*& input_dataset = input_datasets.back();
    threads.push_back(ctx_->StartThread(
        "flat_map_random_access_iterator",
        [this, input_tensors = std::move(input_tensors), &input_dataset,
         &map_func, &status, &mu]() {
          absl::StatusOr<DatasetBase*> dataset =
              MakeInputDataset(std::move(input_tensors), *map_func);
          if (!dataset.ok()) {
            absl::MutexLock l(mu);
            status.Update(dataset.status());
            return;
          }
          input_dataset = *dataset;
        }));
  }
  threads.clear();
  TF_RETURN_IF_ERROR(std::move(status));
  return input_datasets;
}

absl::StatusOr<DatasetBase*> FlatMapRandomAccessHandler::MakeInputDataset(
    std::vector<Tensor> input_tensors,
    const InstantiatedCapturedFunction& map_func) const {
  std::vector<Tensor> mapped_tensors;
  TF_RETURN_IF_ERROR(
      map_func.Run(ctx_.get(), std::move(input_tensors), &mapped_tensors));
  if (!(mapped_tensors.size() == 1 && mapped_tensors[0].dtype() == DT_VARIANT &&
        TensorShapeUtils::IsScalar(mapped_tensors[0].shape()))) {
    return absl::InvalidArgumentError(
        "Flat map function must return a single scalar of dtype DT_VARIANT "
        "representing a dataset.");
  }

  DatasetBase* mapped_dataset = nullptr;
  TF_RETURN_IF_ERROR(
      GetDatasetFromVariantTensor(mapped_tensors[0], &mapped_dataset));
  mapped_dataset->Ref();
  return mapped_dataset;
}
}  // namespace data
}  // namespace tensorflow
