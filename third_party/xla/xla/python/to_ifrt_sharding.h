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

#ifndef XLA_PYTHON_TO_IFRT_SHARDING_H_
#define XLA_PYTHON_TO_IFRT_SHARDING_H_

#include "nanobind/nanobind.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

// Gets `xla::HloSharding` from a JAX Sharding.
xla::HloSharding GetXlaHloSharding(nanobind::handle sharding,
                                   int64_t num_dimensions);

// Gets `xla::ifrt::DeviceList` from a JAX Sharding.
absl::StatusOr<xla::ifrt::DeviceListRef> GetIfrtDeviceList(
    nanobind::handle sharding_py);

// Gets `xla::ifrt::MemoryKind` from a JAX Sharding.
xla::ifrt::MemoryKind GetMemoryKind(nanobind::handle sharding);

// Converts a JAX Sharding into `xla::ifrt::HloSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>> GetIfrtHloSharding(
    nanobind::handle sharding, const xla::ifrt::Shape& shape);

// Converts a JAX Sharding into `xla::ifrt::ConcreteEvenSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>>
GetIfrtConcreteEvenSharding(nanobind::handle sharding, xla::ifrt::DType dtype,
                            const xla::ifrt::Shape& shape);

// Converts a JAX Sharding into `xla::ifrt::ConcreteSharding`.
absl::StatusOr<std::shared_ptr<const xla::ifrt::Sharding>>
GetIfrtConcreteSharding(nanobind::handle sharding,
                        const xla::ifrt::Shape& shape,
                        std::vector<xla::ifrt::Shape> shard_shapes);

}  // namespace xla

#endif  // XLA_PYTHON_TO_IFRT_SHARDING_H_
