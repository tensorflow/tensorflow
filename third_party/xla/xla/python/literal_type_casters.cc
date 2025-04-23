/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/python/literal_type_casters.h"

#include <Python.h>

#include <cstdint>
#include <optional>

#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"  // IWYU pragma: keep
#include "xla/layout.h"
#include "xla/pjrt/exceptions.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/python/lib/core/numpy.h"
#include "xla/xla_data.pb.h"

namespace nb = nanobind;

namespace xla {

std::optional<CastToArrayResult> CastToArray(nb::handle h) {
  auto array =
      nb_numpy_ndarray::ensure(h, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
  auto type_or_status = DtypeToPrimitiveType(array.dtype());
  if (!type_or_status.ok()) {
    throw xla::XlaRuntimeError(type_or_status.status());
  }
  PrimitiveType type = type_or_status.value();

  absl::InlinedVector<int64_t, 4> dims(array.ndim());
  for (int i = 0; i < array.ndim(); ++i) {
    dims[i] = array.shape(i);
  }
  Shape shape = ShapeUtil::MakeShape(type, dims);
  if (array.size() * array.itemsize() != ShapeUtil::ByteSizeOf(shape)) {
    throw xla::XlaRuntimeError(absl::StrCat(
        "Size mismatch for buffer: ", array.size() * array.itemsize(), " vs. ",
        ShapeUtil::ByteSizeOf(shape)));
  }
  return CastToArrayResult{array, static_cast<const char*>(array.data()),
                           shape};
}

}  // namespace xla
