/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_INTERPRETER_TESTUTIL_H_
#define TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_INTERPRETER_TESTUTIL_H_

#include <cstring>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/attribute_span.h"

namespace mlrt {
namespace testing {

class SymbolTable {
 public:
  int Def(absl::string_view name) {
    auto iter = reg_names_.find(name);
    if (iter != reg_names_.end()) {
      return iter->second;
    }

    int& id = reg_names_[name];
    id = next_reg_id_++;

    return id;
  }

  std::vector<int> Def(absl::Span<const std::string> names) {
    return DefOrUse(names,
                    [this](absl::string_view name) { return Def(name); });
  }

  int Use(absl::string_view name) const {
    DCHECK(reg_names_.contains(name));
    return reg_names_.at(name);
  }

  std::vector<int> Use(absl::Span<const std::string> names) {
    return DefOrUse(names,
                    [this](absl::string_view name) { return Use(name); });
  }

  size_t size() const { return reg_names_.size(); }

 private:
  std::vector<int> DefOrUse(
      absl::Span<const std::string> names,
      absl::FunctionRef<int(absl::string_view)> def_or_use) {
    std::vector<int> ids;
    ids.reserve(names.size());
    for (const auto& name : names) {
      ids.push_back(def_or_use(name));
    }
    return ids;
  }

  absl::flat_hash_map<std::string, int> reg_names_;
  int next_reg_id_ = 0;
};

class AttributeTable {
 public:
  explicit AttributeTable(bc::Vector<bc::String>::Constructor attributes_ctor)
      : ctor_(attributes_ctor) {}

  void Add(absl::string_view name, absl::string_view value) {
    handles_[name] = next_id_;
    ctor_.ConstructAt(next_id_++, value);
  }

  void Add(absl::string_view name, const char* value) {
    Add(name, absl::string_view(value));
  }

  void AddInline(absl::string_view name, absl::string_view value) {
    DCHECK_LE(value.size(), sizeof(uint32_t));
    std::memcpy(&handles_[name], value.data(), value.size());
  }

  template <typename T,
            typename std::enable_if_t<
                attribute_internal::kCanAttributeBeInlined<T>, int> = 0>
  void Add(absl::string_view name, T value) {
    AddInline(name, absl::string_view(reinterpret_cast<const char*>(&value),
                                      sizeof(value)));
  }

  template <typename T, typename std::enable_if_t<
                            std::is_trivial_v<T> &&
                                !attribute_internal::kCanAttributeBeInlined<T>,
                            int> = 0>
  void Add(absl::string_view name, T value) {
    Add(name, absl::string_view(reinterpret_cast<const char*>(&value),
                                sizeof(value)));
  }

  uint32_t GetHandle(absl::string_view name) { return handles_.at(name); }

 private:
  bc::Vector<bc::String>::Constructor ctor_;
  int next_id_ = 0;
  absl::flat_hash_map<std::string, uint32_t> handles_;
};

}  // namespace testing
}  // namespace mlrt

#endif  // TENSORFLOW_CORE_TFRT_MLRT_INTERPRETER_INTERPRETER_TESTUTIL_H_
