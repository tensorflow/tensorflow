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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/array.h"
#include "xla/shape.h"
#include "xla/tsl/concurrency/future.h"

namespace tensorflow {
namespace ifrt_serving {

// This class is thread safe.
class IfrtLoadedVariableRegistry {
 public:
  // The key is per variable tensor per device assignment. For single -device
  // program, variables can be loaded on multiple devices with core selection.
  // For SPMD program, we currently assume all devices will be used, so we use
  // vector to make it compatible with SPMD.
  struct Key {
    // We use a vector to make it compatible with SPMD because the order of the
    // devices used for sharding must match the order of the devices used for
    // xla compilation.
    std::vector<int> device_ids;
    std::string input_name;
    xla::HloSharding hlo_sharding;
    // Use pointer instead of value to avoid copy when building the key.
    std::shared_ptr<xla::Shape> shape_on_device;
    template <typename H>
    friend H AbslHashValue(H h, const Key& key) {
      h = H::combine(std::move(h), key.input_name, key.device_ids,
                     key.hlo_sharding);
      if (key.shape_on_device != nullptr) {
        h = H::combine(std::move(h), *key.shape_on_device);
      }
      return h;
    }

    friend bool operator==(const Key& x, const Key& y) {
      return KeyView(x) == KeyView(y);
    }

    std::string ToString() const {
      return absl::StrCat(
          input_name, ":", absl::StrJoin(device_ids, ","), ":",
          hlo_sharding.ToString(), ":",
          (shape_on_device ? shape_on_device->ToString() : "nullptr"));
    }
  };

  // A view of Key that references the data without owning it. This is used for
  // efficient lookups in the LoadedVariable map without constructing a full Key
  // object.
  struct KeyView {
    KeyView(const Key& key)  // NOLINT
        : device_ids(key.device_ids),
          input_name(key.input_name),
          hlo_sharding(key.hlo_sharding),
          shape_on_device(key.shape_on_device) {}

    KeyView(absl::Span<const int> device_ids, absl::string_view input_name,
            const xla::HloSharding& hlo_sharding,
            std::shared_ptr<xla::Shape> shape_on_device)
        : device_ids(device_ids),
          input_name(input_name),
          hlo_sharding(hlo_sharding),
          shape_on_device(std::move(shape_on_device)) {}

    absl::Span<const int> device_ids;
    absl::string_view input_name;
    const xla::HloSharding& hlo_sharding;
    std::shared_ptr<xla::Shape> shape_on_device;

    template <typename H>
    friend H AbslHashValue(H h, const KeyView& key) {
      h = H::combine(std::move(h), key.input_name, key.device_ids,
                     key.hlo_sharding);
      if (key.shape_on_device != nullptr) {
        h = H::combine(std::move(h), *key.shape_on_device);
      }
      return h;
    }

    friend bool operator==(const KeyView& x, const KeyView& y) {
      bool xla_shape_equal = false;
      if (x.shape_on_device == nullptr && y.shape_on_device == nullptr) {
        xla_shape_equal = true;
      } else if (x.shape_on_device != nullptr && y.shape_on_device != nullptr) {
        xla_shape_equal = *x.shape_on_device == *y.shape_on_device;
      }
      return x.input_name == y.input_name && x.device_ids == y.device_ids &&
             x.hlo_sharding == y.hlo_sharding && xla_shape_equal;
    }
  };

  struct KeyEq {
    using is_transparent = void;
    bool operator()(const KeyView& lhs, const KeyView& rhs) const {
      return rhs == lhs;
    }
  };

  struct KeyHash {
    using is_transparent = void;
    size_t operator()(const KeyView& key) const { return absl::HashOf(key); }
  };

  struct LoadedVariable {
    tsl::Future<xla::ifrt::ArrayRef> array;
  };
  using LoadedVariableConstructor =
      absl::AnyInvocable<absl::StatusOr<LoadedVariable>() const>;

  // Tries to register a loaded variable with the given name.
  // Returns an error if the named array does not already exists and
  // loaded_variable_constructor failed to create an array. Note that it returns
  // OK if the named array already exists.
  // loaded_variable_constructor is invoked in the caller thread.
  absl::Status TryRegisterLoadedVariable(
      const Key& key, LoadedVariableConstructor&& loaded_variable_constructor)
      ABSL_LOCKS_EXCLUDED(mutex_);

  absl::StatusOr<LoadedVariable> GetLoadedVariable(KeyView key_view) const
      ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<Key, LoadedVariable, KeyHash, KeyEq> loaded_variable_map_
      ABSL_GUARDED_BY(mutex_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_LOADED_VARIABLE_REGISTRY_H_
