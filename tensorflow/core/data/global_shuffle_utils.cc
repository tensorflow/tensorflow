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
#include "tensorflow/core/data/global_shuffle_utils.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {

IteratorContextWithIndexMapper::IteratorContextWithIndexMapper(
    IteratorContext* ctx, const IteratorBase* iterator)
    : ctx_(ctx) {
  if (ctx_->index_mapper()) {
    IteratorContext::Params params(ctx_);
    params.index_mapper = iterator->GetIndexMapper(ctx_->index_mapper());
    ctx_with_index_mapper_.emplace(params);
  }
}

IteratorContext* IteratorContextWithIndexMapper::Get() {
  return ctx_with_index_mapper_.has_value() ? &ctx_with_index_mapper_.value()
                                            : ctx_;
}

void IteratorContextWithIndexMapper::MergeCheckpoint() {
  if (ctx_with_index_mapper_.has_value()) {
    ctx_->MergeCheckpoint(ctx_with_index_mapper_->checkpoint());
  }
}

absl::Status GlobalShuffleIterator::GetNext(IteratorContext* ctx,
                                            std::vector<Tensor>* out_tensors,
                                            bool* end_of_sequence) {
  if (!ctx->index_mapper()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Trying to get a random element from dataset ", dataset_->DebugString(),
        " which is not globally shuffled."));
  }

  absl::MutexLock l(&mu_);
  TF_ASSIGN_OR_RETURN(int64_t output_index,
                      ctx->index_mapper()(element_count_++));
  absl::Status status =
      dataset_->Get(AnyContext(ctx), output_index, out_tensors);
  if (absl::IsOutOfRange(status)) {
    *end_of_sequence = true;
    return absl::OkStatus();
  }
  TF_RETURN_IF_ERROR(status);
  *end_of_sequence = false;
  return absl::OkStatus();
}

absl::Status GlobalShuffleIterator::Restore(IteratorContext* ctx) {
  if (!ctx->restored_element_count().has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Trying to restore random element count for dataset ",
        dataset_->DebugString(), " which is not globally shuffled."));
  }

  absl::MutexLock l(&mu_);
  element_count_ = *(ctx->restored_element_count());
  return absl::OkStatus();
}

}  // namespace data
}  // namespace tensorflow
