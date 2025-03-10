/* Copyright 2025 The OpenXLA Authors.
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
#ifndef XLA_PJRT_PJRT_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
#define XLA_PJRT_PJRT_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_

#include <array>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_device_description.h"

namespace xla {

class PjRtStreamExecutorDeviceDescription : public PjRtDeviceDescription {
 public:
  explicit PjRtStreamExecutorDeviceDescription(int id, std::string device_kind,
                                               int process_index = 0)
      : id_(id),
        process_index_(process_index),
        device_kind_(std::move(device_kind)) {}

  int id() const override { return id_; }

  int process_index() const override { return process_index_; }

  absl::string_view device_kind() const override { return device_kind_; }

  absl::string_view ToString() const override { return to_string_; }

  absl::string_view DebugString() const override { return debug_string_; }

  absl::Span<int const> coords() const { return absl::MakeSpan(coords_); }

  const absl::flat_hash_map<std::string, PjRtDeviceAttribute>& Attributes()
      const override {
    return attributes_;
  }

  void SetAttributes(
      absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes) {
    attributes_ = std::move(attributes);
  }

  void SetDebugString(std::string debug_string) {
    debug_string_ = std::move(debug_string);
  }

  void SetToString(std::string to_string) { to_string_ = std::move(to_string); }

  void SetCoords(std::array<int, 1> coords) { coords_ = coords; }

 private:
  const int id_;
  const int process_index_;
  const std::string device_kind_;
  std::string debug_string_ = "<unknown SE device>";
  std::string to_string_ = "<unknown SE device>";
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attributes_;
  std::array<int, 1> coords_;
};
}  // namespace xla

#endif  // XLA_PJRT_PJRT_STREAM_EXECUTOR_DEVICE_DESCRIPTION_H_
