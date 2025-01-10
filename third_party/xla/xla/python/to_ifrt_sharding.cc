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
#include <utility>

#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
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
absl::StatusOr<tsl::RCReference<xla::ifrt::DeviceList>> GetIfrtDeviceList(
    nb::handle sharding_py) {
  nb::handle sharding(sharding_py.ptr());
  if (sharding.type().is(jax::NamedSharding::type())) {
    TF_ASSIGN_OR_RETURN(
        auto ns_device_list,
        nb::cast<const jax::NamedSharding*>(sharding)->internal_device_list());
    return ns_device_list->ifrt_device_list();
  } else if (sharding.type().is(jax::SingleDeviceSharding::type())) {
    return nb::cast<const jax::SingleDeviceSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else if (sharding.type().is(jax::PmapSharding::type())) {
    return nb::cast<const jax::PmapSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else if (sharding.type().is(jax::GSPMDSharding::type())) {
    return nb::cast<const jax::GSPMDSharding*>(sharding)
        ->internal_device_list()
        ->ifrt_device_list();
  } else {
    return nb::cast<const jax::PyDeviceList*>(
               sharding.attr("_internal_device_list"))
        ->ifrt_device_list();
  }
}

// Converts a JAX Sharding into `xla::ifrt::HloSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>> GetIfrtHloSharding(
    nb::handle sharding, const xla::ifrt::Shape& shape) {
  TF_ASSIGN_OR_RETURN(tsl::RCReference<xla::ifrt::DeviceList> device_list,
                      GetIfrtDeviceList(sharding));
  xla::HloSharding hlo_sharding =
      GetXlaHloSharding(sharding, shape.dims().size());
  return xla::ifrt::HloSharding::Create(
      std::move(device_list), xla::ifrt::MemoryKind(), std::move(hlo_sharding));
}

// Converts a JAX Sharding into `xla::ifrt::ConcreteEvenSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>>
GetIfrtConcreteEvenSharding(nb::handle sharding, xla::ifrt::DType dtype,
                            const xla::ifrt::Shape& shape) {
  TF_ASSIGN_OR_RETURN(tsl::RCReference<xla::ifrt::DeviceList> device_list,
                      GetIfrtDeviceList(sharding));
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
      std::move(device_list), xla::ifrt::MemoryKind(), shape,
      /*shard_shape=*/std::move(shard_shape));
}

}  // namespace xla
