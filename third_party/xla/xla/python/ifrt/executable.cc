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
#include "xla/python/ifrt/execute_options.pb.h"

namespace xla {
namespace ifrt {

char Executable::ID = 0;
char LoadedExecutable::ID = 0;

absl::StatusOr<xla::ifrt::ExecuteOptionsProto> ExecuteOptions::ToProto() const {
  ExecuteOptionsProto proto;

  proto.set_launch_id(launch_id);
  proto.mutable_non_donatable_input_indices()->Add(
      non_donatable_input_indices.begin(), non_donatable_input_indices.end());
  proto.set_fill_status(fill_status);

  return proto;
}

absl::StatusOr<xla::ifrt::ExecuteOptions> ExecuteOptions::FromProto(
    const xla::ifrt::ExecuteOptionsProto& proto) {
  ExecuteOptions options;

  options.launch_id = proto.launch_id();
  options.non_donatable_input_indices.insert(
      proto.non_donatable_input_indices().begin(),
      proto.non_donatable_input_indices().end());
  options.fill_status = proto.fill_status();

  return options;
}

}  // namespace ifrt
}  // namespace xla
