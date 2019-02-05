/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/renamed_device.h"
#include "absl/memory/memory.h"

namespace tensorflow {

/* static */
std::unique_ptr<Device> RenamedDevice::NewRenamedDevice(
    const string& new_base, Device* underlying, bool owns_underlying,
    bool isolate_session_state) {
  DeviceNameUtils::ParsedName parsed_name;
  CHECK(DeviceNameUtils::ParseFullName(new_base, &parsed_name));
  DeviceNameUtils::ParsedName underlying_parsed_name =
      underlying->parsed_name();
  CHECK(underlying_parsed_name.has_type);
  CHECK(underlying_parsed_name.has_id);
  parsed_name.type = underlying_parsed_name.type;
  parsed_name.id = underlying_parsed_name.id;
  string name = DeviceNameUtils::FullName(parsed_name.job, parsed_name.replica,
                                          parsed_name.task, parsed_name.type,
                                          parsed_name.id);
  DeviceAttributes attributes(underlying->attributes());
  attributes.set_name(name);
  // Call absl::WrapUnique to access private constructor.
  return absl::WrapUnique(new RenamedDevice(
      underlying, attributes, owns_underlying, isolate_session_state));
}

RenamedDevice::RenamedDevice(Device* underlying,
                             const DeviceAttributes& attributes,
                             bool owns_underlying, bool isolate_session_state)
    : Device(underlying->env(), attributes),
      underlying_(underlying),
      owns_underlying_(owns_underlying),
      isolate_session_state_(isolate_session_state) {}

RenamedDevice::~RenamedDevice() {
  if (owns_underlying_) {
    delete underlying_;
  }
}

}  // namespace tensorflow
