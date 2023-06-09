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

#ifndef TENSORFLOW_CORE_DATA_SPLIT_UTILS_H_
#define TENSORFLOW_CORE_DATA_SPLIT_UTILS_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// A class which produces splits for a dataset of size N that can be indexed
// into.
class IndexSplitProvider : public SplitProvider {
 public:
  explicit IndexSplitProvider(int64_t n);
  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  mutex mu_;
  int64_t i_ TF_GUARDED_BY(mu_);
  const int64_t n_;
};

// A SplitProvider which wraps another split provider, but drops all splits
// where `index != shard_index % num_shards`
class ShardingSplitProvider : public SplitProvider {
 public:
  ShardingSplitProvider(int64_t num_shards, int64_t shard_index,
                        std::shared_ptr<SplitProvider> split_provider);

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const int64_t num_shards_;
  const int64_t shard_index_;
  mutex mu_;
  std::shared_ptr<SplitProvider> split_provider_ TF_GUARDED_BY(mu_);
  int64_t num_to_skip_ TF_GUARDED_BY(mu_);
};

// Returns split providers for all sources of the given dataset.
StatusOr<std::vector<std::unique_ptr<SplitProvider>>> GetSplitProviders(
    const DatasetBase* dataset);

// Gets the single split provider from the context, or returns an error if the
// context has zero or multiple split providers. The `dataset` argument is used
// to produce a more useful error message.
StatusOr<std::shared_ptr<SplitProvider>> GetSingleSplitProvider(
    IteratorContext* ctx, const DatasetBase* dataset);

// Creates iterator contexts for datasets inputs. The split providers
// in `ctx` will be divided among the inputs of `dataset`, so that each input
// gets a number of split providers that matches its number of source datasets.
// If no split providers are defined, the contexts will be the same as `ctx`.
StatusOr<std::vector<IteratorContext>> CreateInputIteratorContexts(
    IteratorContext* ctx, const DatasetBase* dataset);
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SPLIT_UTILS_H_
