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
#include "tensorflow/core/data/split_utils.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/types.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kNumToSkip[] = "num_to_skip";
constexpr char kSplitProvider[] = "split_provider";
constexpr char kSlash[] = "/";
constexpr char kIndex[] = "index";
}  // namespace

IndexSplitProvider::IndexSplitProvider(int64_t n) : i_(0), n_(n) {
  VLOG(3) << "Created index split provider with " << n << " splits.";
}

absl::Status IndexSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
  tsl::mutex_lock l(mu_);
  if (i_ >= n_) {
    *end_of_splits = true;
    return absl::OkStatus();
  }
  *end_of_splits = false;
  *split = Tensor(DT_INT64, TensorShape{});
  split->scalar<int64_t>()() = i_++;
  return absl::OkStatus();
}

absl::Status IndexSplitProvider::Reset() {
  tsl::mutex_lock l(mu_);
  i_ = 0;
  return absl::OkStatus();
}

absl::Status IndexSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  tsl::mutex_lock l(mu_);
  return writer->WriteScalar(full_name(kIndex), i_);
}

absl::Status IndexSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  tsl::mutex_lock l(mu_);
  return reader->ReadScalar(full_name(kIndex), &i_);
}

int64_t IndexSplitProvider::Cardinality() const {
  // RandomDataset uses kint64max to simulate infinite splits.
  // See RandomDatasetOp::Dataset::MakeSplitProviders.
  if (n_ == tsl::kint64max) {
    return kInfiniteCardinality;
  }
  return n_;
}

ShardingSplitProvider::ShardingSplitProvider(
    int64_t num_shards, int64_t shard_index,
    std::shared_ptr<SplitProvider> split_provider)
    : num_shards_(num_shards),
      shard_index_(shard_index),
      split_provider_(split_provider),
      num_to_skip_(shard_index_) {}

absl::Status ShardingSplitProvider::GetNext(Tensor* split,
                                            bool* end_of_splits) {
  tsl::mutex_lock l(mu_);
  while (num_to_skip_ > 0) {
    TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
    if (*end_of_splits) {
      return absl::OkStatus();
    }
    num_to_skip_--;
  }
  num_to_skip_ = num_shards_ - 1;
  TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
  return absl::OkStatus();
}

absl::Status ShardingSplitProvider::Reset() {
  tsl::mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Reset());
  num_to_skip_ = shard_index_;
  return absl::OkStatus();
}

absl::Status ShardingSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  tsl::mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Save(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      writer));
  return writer->WriteScalar(full_name(kNumToSkip), num_to_skip_);
}

absl::Status ShardingSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  tsl::mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Restore(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      reader));
  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumToSkip), &num_to_skip_));
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<SplitProvider>> GetSingleSplitProvider(
    IteratorContext* ctx, const DatasetBase* dataset) {
  if (ctx->split_providers().size() != 1) {
    return absl::FailedPreconditionError(
        absl::StrCat("Failed to get single split provider for dataset ",
                     dataset->DebugString(), ". Found ",
                     ctx->split_providers().size(), " split providers"));
  }
  return ctx->split_providers()[0];
}

absl::StatusOr<std::vector<std::unique_ptr<SplitProvider>>> GetSplitProviders(
    const DatasetBase* dataset) {
  std::vector<std::unique_ptr<SplitProvider>> result;
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset->InputDatasets(&inputs));
  for (const auto& input : inputs) {
    std::vector<std::unique_ptr<SplitProvider>> providers;
    TF_RETURN_IF_ERROR(input->MakeSplitProviders(&providers));
    for (auto& provider : providers) {
      result.push_back(std::move(provider));
    }
  }
  return result;
}

absl::StatusOr<std::vector<IteratorContext>> CreateInputIteratorContexts(
    IteratorContext* ctx, const DatasetBase* dataset) {
  std::vector<const DatasetBase*> inputs;
  TF_RETURN_IF_ERROR(dataset->InputDatasets(&inputs));
  std::vector<IteratorContext> result;
  if (ctx->split_providers().empty()) {
    for (int i = 0; i < inputs.size(); ++i) {
      result.emplace_back(ctx);
    }
    return result;
  }
  int64_t num_sources = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]->num_sources() < 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Failed to determine the number of sources for dataset of type ",
          inputs[i]->type_string()));
    }
    num_sources += inputs[i]->num_sources();
  }
  if (num_sources != ctx->split_providers().size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Attempted to feed ", ctx->split_providers().size(),
        " split providers into a dataset with ", num_sources, " sources"));
  }
  int64_t split_provider_index = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    IteratorContext::Params params(ctx);
    params.split_providers.clear();
    for (int j = 0; j < inputs[i]->num_sources(); ++j) {
      params.split_providers.push_back(
          ctx->split_providers()[split_provider_index + j]);
    }
    split_provider_index += inputs[i]->num_sources();
    result.emplace_back(std::move(params));
  }
  return result;
}

}  // namespace data
}  // namespace tensorflow
