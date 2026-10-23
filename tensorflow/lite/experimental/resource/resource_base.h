/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_BASE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_BASE_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "tensorflow/lite/core/c/c_api_types.h"

namespace tflite {
namespace resource {

// ResourceBase is an abstract base class for resources.
/// WARNING: Experimental interface, subject to change.
class ResourceBase {
 public:
  enum class ResourceType {
    kUnknown = 0,
    kResourceVariable = 1,
    kHashTable = 2,
    kInitializationStatus = 3,
  };

  explicit ResourceBase() {}
  virtual ~ResourceBase() {}

  virtual ResourceType GetResourceType() const = 0;

  // Returns true if it is initialized.
  virtual bool IsInitialized() = 0;

  virtual size_t GetMemoryUsage() {
    return 0;
  }  // TODO(b/242603814): Make it pure virtual.
};

/// WARNING: Experimental interface, subject to change.
using ResourceMap =
    std::unordered_map<std::int32_t, std::unique_ptr<ResourceBase>>;

using ResourceIDMap = std::map<std::pair<std::string, std::string>, int>;

// Generic lookup helper with type safety.
template <typename T>
T* GetTypedResource(ResourceMap* resources, int resource_id,
                    ResourceBase::ResourceType expected_type) {
  if (resources == nullptr) {
    return nullptr;
  }
  auto it = resources->find(resource_id);
  if (it != resources->end() && it->second != nullptr &&
      it->second->GetResourceType() == expected_type) {
    return static_cast<T*>(it->second.get());
  }
  return nullptr;
}

// Generic creation helper with type check and factory fallback.
template <typename Factory>
TfLiteStatus CreateTypedResourceIfNotAvailable(
    ResourceMap* resources, int resource_id,
    ResourceBase::ResourceType expected_type, Factory&& factory) {
  if (resources == nullptr) {
    return kTfLiteError;
  }
  auto it = resources->find(resource_id);
  if (it != resources->end()) {
    if (it->second == nullptr ||
        it->second->GetResourceType() != expected_type) {
      return kTfLiteError;
    }
    return kTfLiteOk;
  }
  auto resource = factory();
  if (resource == nullptr) {
    return kTfLiteError;
  }
  resources->emplace(resource_id, std::move(resource));
  return kTfLiteOk;
}

}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_BASE_H_
