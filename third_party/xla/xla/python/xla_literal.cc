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

#include "xla/python/xla_literal.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/dlpack.h"
#include "xla/python/types.h"
#include "xla/shape.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace {

namespace nb = nanobind;

// Helper to convert an xla::Literal to a nanobind::ndarray.
nb::ndarray<> LiteralToNdarray(Literal& obj) {
  const Shape& shape = obj.shape();
  LOG(INFO) << "LiteralToNdarray: " << shape.ToString() << " "
            << shape.has_layout();

  if (!shape.has_layout()) {
    throw XlaRuntimeError(
        "Creating an array is only supported for Literals with a layout.");
  }

  const Layout& layout = shape.layout();

  if (!layout.tiles().empty()) {
    throw XlaRuntimeError(
        "Creating an array from a tiled Literal is not supported.");
  }

  if (!shape.IsArray()) {
    throw XlaRuntimeError(
        "Creating an array is only supported for dense Literals.");
  }

  xla::PrimitiveType primitive_type = shape.element_type();
  nb::dlpack::dtype dtype =
      ValueOrThrow(xla::PrimitiveTypeToNbDLDataType(primitive_type));

  absl::Span<const int64_t> dimensions = shape.dimensions();
  std::vector<size_t> unsigned_dimensions(dimensions.begin(), dimensions.end());
  auto strides = xla::StridesForShape(primitive_type, dimensions, layout);

  return nb::ndarray<>(obj.untyped_data(), unsigned_dimensions.size(),
                       unsigned_dimensions.data(), {}, strides.data(), dtype,
                       nb::device::cpu::value, 0);
}

}  // namespace

void BuildLiteral(nb::module_& m) {
  nb::class_<Literal>(m, "Literal")
      .def(nb::init<const Shape&>())
      .def("__repr__", &Literal::ToString)
      .def(
          "__array__",
          [](std::shared_ptr<Literal> obj, std::optional<nb::object> dtype,
             std::optional<bool> copy) {
            // Provides the interface required by numpy to create a np.ndarray.
            nb::ndarray<nb::numpy> np_array(LiteralToNdarray(*obj));

            if (dtype.has_value()) {
              throw XlaRuntimeError(
                  "Passing of dtype to __array__ not currently supported.");
            }

            if (copy.has_value() && *copy) {
              // when a copy is requested we _must_ return a copy:
              // https://numpy.org/doc/2.1/reference/generated/numpy.ndarray.__array__.html
              return np_array.cast(nb::rv_policy::copy);
            }

            return np_array.cast(nb::rv_policy::reference_internal,
                                 nb::cast(obj));
          },
          nb::arg("dtype").none() = nb::none(),
          nb::arg("copy").none() = nb::none())
      .def("shape", &Literal::shape);
}

}  // namespace xla
