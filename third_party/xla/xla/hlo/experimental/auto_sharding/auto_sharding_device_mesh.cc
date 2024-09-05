/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/experimental/auto_sharding/auto_sharding_device_mesh.h"

#include <cstdint>

#include "absl/types/span.h"
#include "xla/array.h"

namespace xla {
namespace spmd {

void DeviceMesh::SetValues(absl::Span<const int64_t> values) {
  device_array.SetValues(values);
  is_iota = AreValuesIota(values);
}
}  // namespace spmd
}  // namespace xla
