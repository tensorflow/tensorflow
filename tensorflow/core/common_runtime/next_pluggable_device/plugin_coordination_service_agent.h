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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_H_

#include <string>

#include "tensorflow/core/platform/statusor.h"

namespace tsl {
class Status;
}  // namespace tsl
namespace tensorflow {
using tsl::Status;

class PluginCoordinationServiceAgent {
 public:
  PluginCoordinationServiceAgent() = default;
  virtual ~PluginCoordinationServiceAgent() = default;

  virtual bool IsInitialized() const = 0;

  virtual Status InsertKeyValue(const std::string& key,
                                const std::string& value) = 0;

  virtual StatusOr<std::string> GetKeyValue(const std::string& key) = 0;

  virtual Status DeleteKeyValue(const std::string& key) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_PLUGIN_COORDINATION_SERVICE_AGENT_H_
