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
#include "xla/python/to_ifrt_sharding.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/python/py_device_list.h"
#include "xla/python/sharding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace nb = ::nanobind;

// Gets `xla::HloSharding` from a JAX Sharding.
xla::HloSharding GetXlaHloSharding(nb::handle sharding,
                                   int64_t num_dimensions) {
  if (sharding.type().is(nb::handle(jax::GSPMDSharding::type().ptr()))) {
    return nb::cast<jax::GSPMDSharding*>(nb::handle(sharding.ptr()))
        ->hlo_sharding();
  } else {
    return nb::cast<xla::HloSharding>(
        sharding.attr("_to_xla_hlo_sharding")(num_dimensions));
  }
}

// Gets `xla::ifrt::DeviceList` from a JAX Sharding.
absl::StatusOr<xla::ifrt::DeviceListRef> GetIfrtDeviceList(
    nb::handle sharding_py) {
  TF_ASSIGN_OR_RETURN(auto py_device_list, jax::GetPyDeviceList(sharding_py));
  return py_device_list->ifrt_device_list();
}

// Gets `xla::ifrt::MemoryKind` from a JAX Sharding.
xla::ifrt::MemoryKind GetMemoryKind(nb::handle sharding) {
  nb::object py_memory_kind = nb::none();

  // sharding.attr("memory_kind") can crash if sharding was originally created
  // from C++ and casted into a Python Sharding object. Thus, we cast sharding
  // to a C++ type and use C++ `memory_kind()` method, which bypasses any Python
  // attribute access.
  nb::handle type = sharding.type();
  if (type.is(jax::NamedSharding::type())) {
    py_memory_kind =
        nb::cast<const jax::NamedSharding*>(sharding)->memory_kind();
  } else if (type.is(jax::SingleDeviceSharding::type())) {
    py_memory_kind =
        nb::cast<const jax::SingleDeviceSharding*>(sharding)->memory_kind();
  } else if (type.is(jax::GSPMDSharding::type())) {
    py_memory_kind =
        nb::cast<const jax::GSPMDSharding*>(sharding)->memory_kind();
  } else {
    py_memory_kind = sharding.attr("memory_kind");
  }

  if (py_memory_kind.is_none()) {
    return xla::ifrt::MemoryKind();
  }
  return xla::ifrt::MemoryKind(nb::cast<std::string>(py_memory_kind));
}

// Converts a JAX Sharding into `xla::ifrt::HloSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>> GetIfrtHloSharding(
    nb::handle sharding, const xla::ifrt::Shape& shape) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      GetIfrtDeviceList(sharding));
  xla::ifrt::MemoryKind memory_kind = GetMemoryKind(sharding.ptr());
  xla::HloSharding hlo_sharding =
      GetXlaHloSharding(sharding, shape.dims().size());
  return xla::ifrt::HloSharding::Create(
      std::move(device_list), std::move(memory_kind), std::move(hlo_sharding));
}

// Converts a JAX Sharding into `xla::ifrt::ConcreteEvenSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>>
GetIfrtConcreteEvenSharding(nb::handle sharding, xla::ifrt::DType dtype,
                            const xla::ifrt::Shape& shape) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      GetIfrtDeviceList(sharding));
  xla::ifrt::MemoryKind memory_kind = GetMemoryKind(sharding.ptr());
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType xla_primitive_type,
                      xla::ifrt::ToPrimitiveType(dtype));
  // The XLA shape's layout is irrelevant because we only need to know the
  // tile shape, which is independent from the layout.
  xla::Shape xla_shape = xla::ShapeUtil::MakeShapeWithDescendingLayout(
      xla_primitive_type, shape.dims());
  xla::HloSharding hlo_sharding =
      GetXlaHloSharding(sharding, shape.dims().size());
  xla::Shape tile_shape = hlo_sharding.TileShape(xla_shape);
  xla::ifrt::Shape shard_shape(xla::ifrt::Shape::Dimensions(
      tile_shape.dimensions().begin(), tile_shape.dimensions().end()));
  return xla::ifrt::ConcreteEvenSharding::Create(
      std::move(device_list), std::move(memory_kind), shape,
      /*shard_shape=*/std::move(shard_shape));
}

// Converts a JAX Sharding into `xla::ifrt::ConcreteSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>>
GetIfrtConcreteSharding(nb::handle sharding, const xla::ifrt::Shape& shape,
                        std::vector<xla::ifrt::Shape> shard_shapes) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      GetIfrtDeviceList(sharding));
  xla::ifrt::MemoryKind memory_kind = GetMemoryKind(sharding.ptr());
  return xla::ifrt::ConcreteSharding::Create(
      std::move(device_list), std::move(memory_kind), shape,
      /*shard_shapes=*/std::move(shard_shapes));
}

}  // namespace xla
