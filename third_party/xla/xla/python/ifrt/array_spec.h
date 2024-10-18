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

#ifndef XLA_PYTHON_IFRT_ARRAY_SPEC_H_
#define XLA_PYTHON_IFRT_ARRAY_SPEC_H_

#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/python/ifrt/array_spec.pb.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"

namespace xla {
namespace ifrt {

// Specification of an array that groups the static properties of an `Array`
// together. Typically used for describing expected or requested static
// properties of an input/output array of an operation.
struct ArraySpec {
  DType dtype;
  Shape shape;
  std::shared_ptr<const Sharding> sharding;
  // TODO(hyeontaek): Add `layout` once expressing the default layout can be
  // done in a symbolic manner.

  // Constructs `ArraySpec` from `ArraySpecProto`.
  static absl::StatusOr<ArraySpec> FromProto(
      DeviceList::LookupDeviceFunc lookup_device, const ArraySpecProto& proto);

  // Returns a `ArraySpecProto` representation.
  absl::StatusOr<ArraySpecProto> ToProto() const;

  std::string DebugString() const;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_ARRAY_SPEC_H_
