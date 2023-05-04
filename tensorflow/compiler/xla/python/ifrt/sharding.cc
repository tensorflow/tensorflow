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

#include "tensorflow/compiler/xla/python/ifrt/sharding.h"

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/ifrt/index_domain.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace ifrt {

char Sharding::ID = 0;
char SingleDeviceSharding::ID = 0;
char OpaqueSharding::ID = 0;
char ShardingParamSharding::ID = 0;

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << sharding.DebugString();
}

std::shared_ptr<const Sharding> SingleDeviceSharding::Create(Device* device) {
  return std::shared_ptr<const Sharding>(new SingleDeviceSharding(device));
}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
SingleDeviceSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument("Single-device sharding does not support disassembly");
}

StatusOr<std::vector<IndexDomain>> SingleDeviceSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  std::vector<IndexDomain> result;
  result.reserve(1);
  result.push_back(IndexDomain(shape));
  return result;
}

std::string SingleDeviceSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("SingleDeviceSharding(%s)",
                         devices_.front()->ToString());
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(DeviceList devices) {
  return std::shared_ptr<const Sharding>(new OpaqueSharding(
      std::move(devices),
      DisassembleFunc([](const OpaqueSharding& sharding,
                         const Shape& shape) -> StatusOr<std::vector<Shape>> {
        return FailedPrecondition(
            "Using an opaque sharding that disallows disassembly: "
            "sharding=%s; shape=%s",
            sharding.DebugString(), shape.DebugString());
      })));
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(
    DeviceList devices, DisassembleFunc disassemble_func) {
  return std::shared_ptr<const Sharding>(
      new OpaqueSharding(std::move(devices), std::move(disassemble_func)));
}

OpaqueSharding::DisassembleFunc OpaqueSharding::MakeDisassembleFuncFromShapes(
    std::vector<Shape> shapes) {
  // Capture shapes in a shared_ptr so that the disassemble function can be
  // copied cheaply.
  return DisassembleFunc(
      [shapes = std::make_shared<std::vector<Shape>>(std::move(shapes))](
          const OpaqueSharding&, const Shape&) -> StatusOr<std::vector<Shape>> {
        return *shapes;
      });
}

OpaqueSharding::OpaqueSharding(DeviceList devices,
                               DisassembleFunc disassemble_func)
    : llvm::RTTIExtends<OpaqueSharding, Sharding>(std::move(devices)),
      disassemble_func_(std::move(disassemble_func)) {}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
OpaqueSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto shapes, disassemble_func_(*this, shape));
  if (shapes.size() != devices_.size()) {
    return FailedPrecondition(
        "DisassembleFunc returned an incorrect number of shapes");
  }
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    result.push_back(
        {std::move(shapes[i]), SingleDeviceSharding::Create(devices_[i])});
  }
  return result;
}

StatusOr<std::vector<IndexDomain>> OpaqueSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument(
      "OpaqueSharding does not have index domain information");
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "OpaqueSharding(%s)",
      absl::StrJoin(devices_, ",", [](std::string* out, const Device* device) {
        absl::StrAppend(out, device->ToString());
      }));
}

StatusOr<std::shared_ptr<const Sharding>> ShardingParamSharding::Create(
    ShardingParam sharding_param, DeviceList devices) {
  int64_t device_count =
      absl::c_accumulate(sharding_param.minor_to_major().axis_sizes, 1,
                         std::multiplies<int64_t>());
  if (device_count != devices.size()) {
    return FailedPrecondition(
        "Device counts don't match. From ShardingParam %d vs from DeviceList "
        "%d",
        device_count, devices.size());
  }
  return std::shared_ptr<const Sharding>(
      new ShardingParamSharding(std::move(sharding_param), std::move(devices)));
}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
ShardingParamSharding::Disassemble(const Shape& shape) const {
  DCHECK(this);
  if (shape.dims().size() != sharding_param_.dim_shards().size()) {
    return FailedPrecondition(
        "Ranks don't match. From Shape %d vs from ShardingParam %d",
        shape.dims().size(), sharding_param_.dim_shards().size());
  }

  std::vector<int64_t> dims;
  dims.reserve(shape.dims().size());
  for (const auto [dim, dim_shards] :
       llvm::zip(shape.dims(), sharding_param_.dim_shards())) {
    if (dim % dim_shards != 0) {
      return FailedPrecondition(
          "Uneven shard is not supported. dim: %d, dim_shards: %d", dim,
          dim_shards);
    }
    dims.push_back(dim / dim_shards);
  }
  Shape local_shape(dims);

  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  for (Device* device : devices_) {
    result.push_back({local_shape, SingleDeviceSharding::Create(device)});
  }

  return result;
}

StatusOr<std::vector<IndexDomain>> ShardingParamSharding::IndexDomains(
    const Shape& shape) const {
  DCHECK(this);
  return Unimplemented("TODO(b/271129892)");
}

std::string ShardingParamSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "ShardingParamSharding(%s, devices: %s)", sharding_param_.DebugString(),
      absl::StrJoin(devices_, ",", [](std::string* out, const Device* device) {
        absl::StrAppend(out, device->ToString());
      }));
}

}  // namespace ifrt
}  // namespace xla
