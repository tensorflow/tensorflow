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

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace ifrt {

char Sharding::ID = 0;
char SingleDeviceSharding::ID = 0;
char OpaqueSharding::ID = 0;

std::ostream& operator<<(std::ostream& os, const Sharding& sharding) {
  return os << sharding.DebugString();
}

std::shared_ptr<const Sharding> SingleDeviceSharding::Create(Device* device) {
  return std::shared_ptr<const Sharding>(new SingleDeviceSharding(device));
}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
SingleDeviceSharding::Explode(const Shape& shape) const {
  DCHECK(this);
  return InvalidArgument("Single-device sharding does not support explosion");
}

std::string SingleDeviceSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat("SingleDeviceSharding(%s)",
                         devices_.front()->ToString());
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(DeviceList devices) {
  return std::shared_ptr<const Sharding>(new OpaqueSharding(
      std::move(devices),
      ExplodeFunc([](const OpaqueSharding& sharding,
                     const Shape& shape) -> StatusOr<std::vector<Shape>> {
        return FailedPrecondition(
            "Using an opaque sharding that disallows explosion: "
            "sharding=%s; shape=%s",
            sharding.DebugString(), shape.DebugString());
      })));
}

std::shared_ptr<const Sharding> OpaqueSharding::Create(
    DeviceList devices, ExplodeFunc explode_func) {
  return std::shared_ptr<const Sharding>(
      new OpaqueSharding(std::move(devices), std::move(explode_func)));
}

OpaqueSharding::ExplodeFunc OpaqueSharding::MakeExplodeFuncFromShapes(
    std::vector<Shape> shapes) {
  // Capture shapes in a shared_ptr so that the explode function can be copied
  // cheaply.
  return ExplodeFunc(
      [shapes = std::make_shared<std::vector<Shape>>(std::move(shapes))](
          const OpaqueSharding&, const Shape&) -> StatusOr<std::vector<Shape>> {
        return *shapes;
      });
}

OpaqueSharding::OpaqueSharding(DeviceList devices, ExplodeFunc explode_func)
    : llvm::RTTIExtends<OpaqueSharding, Sharding>(std::move(devices)),
      explode_func_(std::move(explode_func)) {}

StatusOr<std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>>>
OpaqueSharding::Explode(const Shape& shape) const {
  DCHECK(this);
  TF_ASSIGN_OR_RETURN(auto shapes, explode_func_(*this, shape));
  if (shapes.size() != devices_.size()) {
    return FailedPrecondition(
        "ExplodeFunc returned an incorrect number of shapes");
  }
  std::vector<std::pair<Shape, std::shared_ptr<const Sharding>>> result;
  result.reserve(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    result.push_back(
        {std::move(shapes[i]), SingleDeviceSharding::Create(devices_[i])});
  }
  return result;
}

std::string OpaqueSharding::DebugString() const {
  DCHECK(this);
  return absl::StrFormat(
      "OpaqueSharding(%s)",
      absl::StrJoin(devices_, ",", [](std::string* out, const Device* device) {
        absl::StrAppend(out, device->ToString());
      }));
}

}  // namespace ifrt
}  // namespace xla
