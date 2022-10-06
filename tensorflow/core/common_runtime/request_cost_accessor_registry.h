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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_ACCESSOR_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_ACCESSOR_REGISTRY_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/request_cost_accessor.h"

namespace tensorflow {

// TODO(b/185852990): Create a template Registry that allows registering
// different types (e.g  RequestCostAccessor, CostMeasurement).
//
// RequestCostAccessorRegistry allows to
// - register a RequestCostAccessor type to the global map
// - create an instance of registered RequestCostAccessor.
class RequestCostAccessorRegistry {
 public:
  // Creates an instance of registered RequestCostAccessor by name. If the named
  // RequestCostAccessor is not registered yet, returns nullptr.
  static std::unique_ptr<RequestCostAccessor> CreateByNameOrNull(
      absl::string_view name);

  using Creator = std::function<std::unique_ptr<RequestCostAccessor>()>;

  // Registers a RequestCostAccessor type to the global map. Registering
  // different types of RequestCostAccessor with the same name is prohibited.
  static void RegisterRequestCostAccessor(absl::string_view name,
                                          Creator creator);
};

// Registers a RequestCostAccessor type to the global map. Registering different
// types of RequestCostAccessor with the same name is prohibited.
class RequestCostAccessorRegistrar {
 public:
  explicit RequestCostAccessorRegistrar(
      absl::string_view name, RequestCostAccessorRegistry::Creator creator) {
    RequestCostAccessorRegistry::RegisterRequestCostAccessor(
        name, std::move(creator));
  }
};

#define REGISTER_REQUEST_COST_ACCESSOR(name, MyRequestCostAccessorClass) \
  namespace {                                                            \
  static ::tensorflow::RequestCostAccessorRegistrar                      \
      MyRequestCostAccessorClass##_registrar((name), [] {                \
        return std::make_unique<MyRequestCostAccessorClass>();          \
      });                                                                \
  }  // namespace

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_REQUEST_COST_ACCESSOR_REGISTRY_H_
