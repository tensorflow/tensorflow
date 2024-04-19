/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/common/pjrt_client_factory_registry.h"

#include <memory>
#include <string>
#include <utility>

#include "tsl/platform/statusor.h"

namespace xla {
PjrtClientFactoryRegistry& PjrtClientFactoryRegistry::Get() {
  static PjrtClientFactoryRegistry* kInstance = new PjrtClientFactoryRegistry();
  return *kInstance;
}

tensorflow::InitOnStartupMarker
PjrtClientFactoryRegistry::RegisterPjrtClientFactory(
    const tsl::DeviceType& device_type,
    const PjrtClientFactory& client_factory) {
  tensorflow::mutex_lock l(mu_);
  const std::string& device_type_str = device_type.type_string();
  if (registry_.find(device_type_str) != registry_.end()) {
    LOG(ERROR) << "Duplicate device type " << device_type_str;
  }
  registry_.emplace(device_type_str, client_factory);
  return {};
}

absl::StatusOr<std::unique_ptr<PjRtClient>>
PjrtClientFactoryRegistry::GetPjrtClient(
    const tsl::DeviceType& device_type,
    const PjrtClientFactoryOptions& options) {
  tensorflow::tf_shared_lock l(mu_);
  const std::string& device_type_str = device_type.type_string();
  const auto client_factory_it = registry_.find(device_type_str);
  if (client_factory_it == registry_.end()) {
    std::string error_msg;
    absl::StrAppend(&error_msg, " The PJRT client factory of `",
                    device_type_str,
                    "` is not registered, available client factory: [");
    for (const auto& [device_name, ignored_func] : registry_) {
      absl::StrAppend(&error_msg, "`", device_name, "`, ");
    }
    absl::StrAppend(&error_msg,
                    "]. Did you forget to link with the appropriate "
                    "`pjrt_*_client_registration` library?");
    return tsl::errors::NotFound(error_msg);
  }
  return client_factory_it->second(options);
}
}  // namespace xla
