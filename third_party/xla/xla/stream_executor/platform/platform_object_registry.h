/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_OBJECT_REGISTRY_H_
#define XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_OBJECT_REGISTRY_H_

#include <any>
#include <functional>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform/initialize.h"  // IWYU pragma: keep
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

// This is a global registry for platform-specific instances of arbitrary
// types.
//
// The registry is keyed by a Trait, which is a struct that defines the
// object's Type.
//
// Trait example:
//
//   struct MyTrait {
//     using Type = std::function<int(float, float)>;
//   };
//
// The registry is thread-safe. Registered objects are immutable and cannot be
// overwritten. When requesting an instance a copy is returned, therefore only
// copyable types are supported.
//
// Use the macro STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY to register a
// kernel during application initialization.
class PlatformObjectRegistry {
 public:
  // Returns a reference to the process-wide instance of the registry.
  static PlatformObjectRegistry& GetGlobalRegistry();

  // Registers an object `obj` in the registry. This function is thread-safe.
  template <typename Trait, typename Other>
  absl::Status RegisterObject(Platform::Id platform_id, Other&& obj) {
    return RegisterObject(
        typeid(Trait), platform_id,
        std::make_any<typename Trait::Type>(std::forward<Other>(obj)));
  }

  template <typename Trait>
  absl::StatusOr<std::reference_wrapper<const typename Trait::Type>> FindObject(
      Platform::Id platform_id) {
    TF_ASSIGN_OR_RETURN(const std::any& obj,
                        FindObject(typeid(Trait), platform_id));
    return std::any_cast<const typename Trait::Type&>(obj);
  }

 private:
  absl::Status RegisterObject(const std::type_info& type,
                              Platform::Id platform_id, std::any object);

  absl::StatusOr<std::reference_wrapper<const std::any>> FindObject(
      const std::type_info& type, Platform::Id platform_id) const;

  mutable absl::Mutex mutex_;
  using RegistryKey = std::tuple<std::type_index, Platform::Id>;
  absl::flat_hash_map<RegistryKey, std::any> objects_ ABSL_GUARDED_BY(mutex_);
};

#define STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(identifier, Trait,   \
                                                   platform_id, object) \
  static void RegisterObject##identifier##Impl() {                      \
    absl::Status result =                                               \
        ::stream_executor::PlatformObjectRegistry::GetGlobalRegistry()  \
            .RegisterObject<Trait>(platform_id, object);                \
    if (!result.ok()) {                                                 \
      LOG(FATAL) << "Failed to register object: " << result;            \
    }                                                                   \
  }                                                                     \
  STREAM_EXECUTOR_REGISTER_MODULE_INITIALIZER(                          \
      RegisterObject##identifier, RegisterObject##identifier##Impl());

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_PLATFORM_PLATFORM_OBJECT_REGISTRY_H_
