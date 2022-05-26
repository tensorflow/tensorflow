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

#include "tensorflow/compiler/xla/python/status_casters_util.h"

#include <string>

namespace xla {
namespace status_casters_util {

namespace {

const char kStatusPayloadUrl[] = "xla.status_casters_util.function";

absl::Cord SerializeFunctionPointer(FunctionPtr fn) {
  return absl::Cord(
      absl::string_view(reinterpret_cast<const char*>(&fn), sizeof(fn)));
}

FunctionPtr DeserializeFunctionPointer(const absl::Cord& payload) {
  return *reinterpret_cast<FunctionPtr*>(
      const_cast<char*>(std::string(payload).data()));
}

}  // namespace

void SetFunctionPointerAsPayload(xla::Status& status, FunctionPtr fn) {
  status.SetPayload(kStatusPayloadUrl, SerializeFunctionPointer(fn));
}

std::optional<FunctionPtr> GetFunctionPointerFromPayload(
    const xla::Status& status) {
  std::optional<absl::Cord> payload = status.GetPayload(kStatusPayloadUrl);

  if (!payload.has_value()) {
    return std::nullopt;
  }

  return DeserializeFunctionPointer(payload.value());
}

}  // namespace status_casters_util
}  // namespace xla
