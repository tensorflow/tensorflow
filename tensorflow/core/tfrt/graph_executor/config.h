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
#ifndef TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_CONFIG_H_
#define TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_CONFIG_H_

#include <string>

#include "google/protobuf/any.pb.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/tfrt/graph_executor/config.pb.h"

namespace tensorflow {
namespace tfrt_stub {

// The helper class for building RuntimeConfigProto and retrieving configs of
// certain types from the RuntimeConfigProto.
class RuntimeConfig {
 public:
  RuntimeConfig() = default;

  static absl::StatusOr<RuntimeConfig> CreateFromProto(
      RuntimeConfigProto proto);

  template <typename ConcreteProto>
  absl::Status Add(const ConcreteProto& config) {
    const auto& full_name = config.GetDescriptor()->full_name();
    if (map_.contains(full_name)) {
      return absl::AlreadyExistsError(
          absl::StrCat(full_name, " already exists in ModelConfig."));
    }

    size_t id = proto_.config_size();
    if (!proto_.add_config()->PackFrom(config)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Failed to pack proto to Any: ", full_name));
    }
    map_[full_name] = id;
    return absl::OkStatus();
  }

  template <typename ConcreteProto>
  absl::StatusOr<ConcreteProto> Get() const {
    const auto& full_name = ConcreteProto::GetDescriptor()->full_name();
    auto iter = map_.find(full_name);

    if (iter == map_.end()) {
      return absl::NotFoundError(
          absl::StrCat(full_name, " not found in ModelConfig."));
    }

    ConcreteProto config;
    if (!proto_.config().at(iter->second).UnpackTo(&config)) {
      return absl::DataLossError(
          absl::StrCat("Failed to unpack proto: ", full_name));
    }
    return config;
  }

  const RuntimeConfigProto& ToProto() const { return proto_; }

 private:
  RuntimeConfigProto proto_;
  absl::flat_hash_map<std::string, size_t> map_;
};

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_GRAPH_EXECUTOR_CONFIG_H_
