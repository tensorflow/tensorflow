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

#include <optional>

#include "tensorflow/core/framework/dataset.h"

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

}  // namespace data
}  // namespace tensorflow
