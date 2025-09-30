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

#ifndef XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_SDY_H_
#define XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_SDY_H_

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

// LINT.IfChange

// From third_party/openxla/shardy/src/shardy/dialect/sdy/ir/attrs.td
// Info about how a sub-axis is derived from the full axis.
struct SubAxisInfo {
  int64_t pre_size;
  int64_t size;

  bool operator==(const SubAxisInfo& other) const {
    return pre_size == other.pre_size && size == other.size;
  }

  template <typename H>
  friend H AbslHashValue(H h, const SubAxisInfo& c) {
    return H::combine(std::move(h), c.pre_size, c.size);
  }
};

// From third_party/openxla/shardy/src/shardy/dialect/sdy/ir/attrs.td
// Reference to either a full axis or a split sub-axis.
class AxisRef {
 public:
  explicit AxisRef(std::string name,
                   std::optional<SubAxisInfo> sub_axis_info = std::nullopt)
      : name_(std::move(name)), sub_axis_info_(std::move(sub_axis_info)) {}

  const std::string& name() const { return name_; }
  const std::optional<SubAxisInfo>& sub_axis_info() const {
    return sub_axis_info_;
  }

  bool operator==(const AxisRef& other) const {
    return name_ == other.name_ && sub_axis_info_ == other.sub_axis_info_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const AxisRef& c) {
    return H::combine(std::move(h), c.name_, c.sub_axis_info_);
  }

 private:
  std::string name_;
  std::optional<SubAxisInfo> sub_axis_info_;
};

// From third_party/openxla/shardy/src/shardy/dialect/sdy/ir/attrs.td
// Named axis in a mesh.
struct MeshAxis {
  std::string name;
  int64_t size;

  bool operator==(const MeshAxis& other) const {
    return name == other.name && size == other.size;
  }

  template <typename H>
  friend H AbslHashValue(H h, const MeshAxis& c) {
    return H::combine(std::move(h), c.name, c.size);
  }
};

// From third_party/openxla/shardy/src/shardy/dialect/sdy/ir/attrs.td
// Mesh of axes and a list of devices.
class Mesh {
 public:
  explicit Mesh(std::vector<MeshAxis> axes,
                std::optional<std::vector<int64_t>> device_ids = std::nullopt)
      : axes_(std::move(axes)), device_ids_(std::move(device_ids)) {}

  const std::vector<MeshAxis>& axes() const { return axes_; }
  const std::optional<std::vector<int64_t>>& device_ids() const {
    return device_ids_;
  }

  bool operator==(const Mesh& other) const {
    return axes_ == other.axes_ && device_ids_ == other.device_ids_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Mesh& c) {
    return H::combine(std::move(h), c.axes_, c.device_ids_);
  }

 private:
  std::vector<MeshAxis> axes_;
  std::optional<std::vector<int64_t>> device_ids_;
};

// LINT.ThenChange(//tensorflow/compiler/xla/hlo/ir/collective_device_list.h)

}  // namespace xla

#endif  // XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_SDY_H_
