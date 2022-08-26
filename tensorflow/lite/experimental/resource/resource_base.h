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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace tflite {
namespace resource {

// ResourceBase is an abstract base class for resources.
/// WARNING: Experimental interface, subject to change.
class ResourceBase {
 public:
  explicit ResourceBase() {}
  virtual ~ResourceBase() {}

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

}  // namespace resource
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_RESOURCE_RESOURCE_BASE_H_
