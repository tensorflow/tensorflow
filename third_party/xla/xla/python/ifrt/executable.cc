/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/executable.h"

#include "absl/status/statusor.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/execute_options.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

char Executable::ID = 0;
char LoadedExecutable::ID = 0;

absl::StatusOr<ExecuteOptionsProto> ExecuteOptions::ToProto() const {
  ExecuteOptionsProto proto;

  proto.set_launch_id(launch_id);
  proto.mutable_non_donatable_input_indices()->Add(
      non_donatable_input_indices.begin(), non_donatable_input_indices.end());
  proto.set_fill_status(fill_status);
  if (custom_options.has_value()) {
    *proto.mutable_custom_options() = custom_options->ToProto();
  }

  return proto;
}

absl::StatusOr<ExecuteOptions> ExecuteOptions::FromProto(
    const ExecuteOptionsProto& proto) {
  ExecuteOptions options;

  options.launch_id = proto.launch_id();
  options.non_donatable_input_indices.insert(
      proto.non_donatable_input_indices().begin(),
      proto.non_donatable_input_indices().end());
  options.fill_status = proto.fill_status();
  if (proto.has_custom_options()) {
    TF_ASSIGN_OR_RETURN(options.custom_options,
                        AttributeMap::FromProto(proto.custom_options()));
  }
  return options;
}

}  // namespace ifrt
}  // namespace xla
