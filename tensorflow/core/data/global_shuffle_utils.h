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
#ifndef TENSORFLOW_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_
#define TENSORFLOW_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_

#include <optional>

#include "tensorflow/core/framework/dataset.h"

namespace tensorflow {
namespace data {

// Builds and selects the `IteratorContext` to use based on whether the dataset
// is globally shuffled.
//
// Example usage in `Iterator::GetNextInternal`:
//
// ```
// IteratorContextWithIndexMapper ctx_with_index_mapper(ctx, this);
// TF_RETURN_IF_ERROR(input_impl_->GetNext(
//     ctx_with_index_mapper.Get(), out_tensors, end_of_sequence));
// ctx_with_index_mapper.MergeCheckpoint();
// ```
//
// The iterator should also implement `GetIndexMapper` if it needs to customize
// the index mapping behavior.
class IteratorContextWithIndexMapper {
 public:
  // Caller keeps ownership of both pointers.
  explicit IteratorContextWithIndexMapper(IteratorContext* ctx,
                                          const IteratorBase* iterator);
  virtual ~IteratorContextWithIndexMapper() = default;
  IteratorContextWithIndexMapper(const IteratorContextWithIndexMapper&) =
      delete;
  IteratorContextWithIndexMapper& operator=(
      const IteratorContextWithIndexMapper&) = delete;

  IteratorContext* Get();
  void MergeCheckpoint();

 private:
  IteratorContext* ctx_;
  std::optional<IteratorContext> ctx_with_index_mapper_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_GLOBAL_SHUFFLE_UTILS_H_
