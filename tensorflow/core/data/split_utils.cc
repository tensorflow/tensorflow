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

#include <functional>
#include <string>
#include <utility>

#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kNumToSkip[] = "num_to_skip";
constexpr char kSplitProvider[] = "split_provider";
constexpr char kSlash[] = "/";
constexpr char kIndex[] = "index";
}  // namespace

IndexSplitProvider::IndexSplitProvider(int64_t n) : i_(0), n_(n) {}

Status IndexSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
  mutex_lock l(mu_);
  if (i_ >= n_) {
    *end_of_splits = true;
    return Status::OK();
  }
  *end_of_splits = false;
  *split = Tensor(DT_INT64, TensorShape{});
  split->scalar<int64_t>()() = i_++;
  return Status::OK();
}

Status IndexSplitProvider::Reset() {
  mutex_lock l(mu_);
  i_ = 0;
  return Status::OK();
}

Status IndexSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  mutex_lock l(mu_);
  return writer->WriteScalar(full_name(kIndex), i_);
}

Status IndexSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  mutex_lock l(mu_);
  return reader->ReadScalar(full_name(kIndex), &i_);
}

ShardingSplitProvider::ShardingSplitProvider(
    int64_t num_shards, int64_t shard_index,
    std::shared_ptr<SplitProvider> split_provider)
    : num_shards_(num_shards),
      shard_index_(shard_index),
      split_provider_(split_provider),
      num_to_skip_(shard_index_) {}

Status ShardingSplitProvider::GetNext(Tensor* split, bool* end_of_splits) {
  mutex_lock l(mu_);
  while (num_to_skip_ > 0) {
    TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
    if (*end_of_splits) {
      return Status::OK();
    }
    num_to_skip_--;
  }
  num_to_skip_ = num_shards_ - 1;
  TF_RETURN_IF_ERROR(split_provider_->GetNext(split, end_of_splits));
  return Status::OK();
}

Status ShardingSplitProvider::Reset() {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Reset());
  num_to_skip_ = shard_index_;
  return Status::OK();
}

Status ShardingSplitProvider::Save(
    std::function<std::string(std::string)> full_name,
    IteratorStateWriter* writer) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Save(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      writer));
  return writer->WriteScalar(full_name(kNumToSkip), num_to_skip_);
}

Status ShardingSplitProvider::Restore(
    std::function<std::string(std::string)> full_name,
    IteratorStateReader* reader) {
  mutex_lock l(mu_);
  TF_RETURN_IF_ERROR(split_provider_->Restore(
      [&](const std::string& key) {
        return full_name(absl::StrCat(kSplitProvider, kSlash, key));
      },
      reader));
  TF_RETURN_IF_ERROR(reader->ReadScalar(full_name(kNumToSkip), &num_to_skip_));
  return Status::OK();
}

StatusOr<std::shared_ptr<SplitProvider>> GetSingleSplitProvider(
    IteratorContext* ctx, const DatasetBase* dataset) {
  if (ctx->split_providers().size() != 1) {
    return errors::FailedPrecondition(
        "Failed to get single split provider for dataset ",
        dataset->DebugString(), ". Found ", ctx->split_providers().size(),
        " split providers");
  }
  return ctx->split_providers()[0];
}

StatusOr<std::vector<std::unique_ptr<SplitProvider>>> GetSplitProviders(
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

StatusOr<std::vector<IteratorContext>> CreateInputIteratorContexts(
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
      return errors::FailedPrecondition(
          "Failed to determine the number of sources for dataset of type ",
          inputs[i]->type_string());
    }
    num_sources += inputs[i]->num_sources();
  }
  if (num_sources != ctx->split_providers().size()) {
    return errors::FailedPrecondition(
        "Attempted to feed ", ctx->split_providers().size(),
        " split providers into a dataset with ", num_sources, " sources");
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
