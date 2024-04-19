/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/initializable_lookup_table.h"

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lookup {

Status InitializableLookupTable::Find(OpKernelContext* ctx, const Tensor& keys,
                                      Tensor* values,
                                      const Tensor& default_value) {
  if (!is_initialized()) {
    return errors::FailedPrecondition("Table not initialized.");
  }
  // Do not let the use migrate before the check;  table is used without
  // a lock by the readers.
  std::atomic_thread_fence(std::memory_order_acquire);
  return DoFind(keys, values, default_value);
}

Status InitializableLookupTable::ImportValues(OpKernelContext* ctx,
                                              const Tensor& keys,
                                              const Tensor& values) {
  lookup::KeyValueTensorIterator iter(&keys, &values);
  auto serializer = std::make_unique<InitializerSerializer>(
      [keys, values](GraphDefBuilder* builder, Node* table, Node** out) {
        Node* keys_node =
            ops::SourceOp("Const", builder->opts()
                                       .WithAttr("dtype", keys.dtype())
                                       .WithAttr("value", keys));
        Node* values_node =
            ops::SourceOp("Const", builder->opts()
                                       .WithAttr("dtype", values.dtype())
                                       .WithAttr("value", values));
        Node* import_table =
            ops::TernaryOp("LookupTableImportV2", table, keys_node, values_node,
                           builder->opts()
                               .WithAttr("Tin", keys.dtype())
                               .WithAttr("Tout", values.dtype()));
        *out = ops::UnaryOp("Identity", table,
                            builder->opts().WithControlInput(import_table));
        return absl::OkStatus();
      });

  return Initialize(iter, std::move(serializer));
}

Status InitializableLookupTable::Initialize(InitTableIterator& iter) {
  return Initialize(iter, /*serializer=*/nullptr);
}

Status InitializableLookupTable::Initialize(
    InitTableIterator& iter,
    std::unique_ptr<InitializerSerializer> serializer) {
  if (!iter.Valid()) {
    return iter.status();
  }
  TF_RETURN_IF_ERROR(
      CheckKeyAndValueTensorsForInsert(iter.keys(), iter.values()));

  mutex_lock l(mu_);
  if (is_initialized()) {
    bool result;
    TF_RETURN_IF_ERROR(AreEntriesSame(iter, &result));
    // If the table is already initialized, we make sure that the entries in the
    // table are the same that we want to initialize the table with.
    if (!result) {
      return errors::FailedPrecondition(
          "Table was already initialized with "
          "different data.");
    } else {
      return absl::OkStatus();
    }
  }
  TF_RETURN_IF_ERROR(DoLazyPrepare([&iter]() { return iter.total_size(); }));
  while (iter.Valid()) {
    TF_RETURN_IF_ERROR(DoInsert(iter.keys(), iter.values()));
    iter.Next();
  }
  if (!errors::IsOutOfRange(iter.status())) {
    return iter.status();
  }

  initializer_serializer_ = std::move(serializer);
  is_initialized_.store(true, std::memory_order_release);
  return absl::OkStatus();
}

Status InitializableLookupTable::AreEntriesSame(const InitTableIterator& iter,
                                                bool* result) {
  *result = static_cast<size_t>(iter.total_size()) == size();
  return absl::OkStatus();
}

}  // namespace lookup
}  // namespace tensorflow
