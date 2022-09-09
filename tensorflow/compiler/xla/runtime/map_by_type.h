/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_RUNTIME_MAP_BY_TYPE_H_
#define TENSORFLOW_COMPILER_XLA_RUNTIME_MAP_BY_TYPE_H_

#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "tensorflow/compiler/xla/runtime/type_id.h"

namespace xla {
namespace runtime {

// An optimized map container for storing pointers of different types.
//
// Example:
//
//   PtrMapByType<IdSet> map;
//
//   int32_t i32 = 0;
//   int64_t i64 = 1;
//
//   map.insert(&i32);
//   map.insert(&i64);
//
//   assert(map.contains<int32_t*>());
//   assert(map.contains<int64_t*>());
//
template <typename IdSet>
class PtrMapByType {
 public:
  template <typename T>
  T* insert(T* value) {
    size_t id = GetDenseTypeId<T>();
    if (id >= data_.size()) {
      data_.resize(id + 1);
    }
    data_[id] = const_cast<std::decay_t<T>*>(value);
    return value;
  }

  template <typename... Ts>
  void insert_all(Ts*... values) {
    static constexpr size_t n = sizeof...(Ts);
    if (n == 0) {
      return;
    }

    // Resize the `data_` to prepare the storage for inserted values.
    std::array<size_t, n> ids = {GetDenseTypeId<Ts>()...};
    data_.resize(1 + *std::max_element(ids.begin(), ids.end()), nullptr);

    (insert<Ts>(values), ...);
  }

  template <typename T>
  T* get() const {
    size_t id = GetDenseTypeId<T>();
    assert(id < data_.size());
    return reinterpret_cast<T*>(data_[id]);
  }

  template <typename T>
  T* getIfExists() const {
    size_t id = GetDenseTypeId<T>();
    return LLVM_LIKELY(id < data_.size()) ? reinterpret_cast<T*>(data_[id])
                                          : nullptr;
  }

  template <typename T>
  bool contains() const {
    size_t id = GetDenseTypeId<T>();
    return id < data_.size() && data_[id] != nullptr;
  }

 private:
  template <typename T>
  static size_t GetDenseTypeId() {
    return DenseTypeId<IdSet>::template get<T>();
  }

  llvm::SmallVector<void*> data_;
};

}  // namespace runtime
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RUNTIME_MAP_BY_TYPE_H_
