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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_SPLIT_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SPLIT_UTILS_H_

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// A class which produces splits for a dataset of size N that can be indexed
// into.
class IndexSplitProvider : public SplitProvider {
 public:
  explicit IndexSplitProvider(int64 n);
  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  mutex mu_;
  int64 i_ TF_GUARDED_BY(mu_);
  const int64 n_;
};

// A SplitProvider which wraps another split provider, but drops all splits
// where `index != shard_index % num_shards`
class ShardingSplitProvider : public SplitProvider {
 public:
  ShardingSplitProvider(int64 num_shards, int64 shard_index,
                        std::shared_ptr<SplitProvider> split_provider);

  Status GetNext(Tensor* split, bool* end_of_splits) override;
  Status Reset() override;
  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override;
  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override;

 private:
  const int64 num_shards_;
  const int64 shard_index_;
  mutex mu_;
  std::shared_ptr<SplitProvider> split_provider_ TF_GUARDED_BY(mu_);
  int64 num_to_skip_ TF_GUARDED_BY(mu_);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SPLIT_UTILS_H_
