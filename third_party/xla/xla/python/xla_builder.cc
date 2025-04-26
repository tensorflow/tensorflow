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

#include "xla/hlo/builder/xla_builder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"  // IWYU pragma: keep
#include "nanobind/stl/string.h"  // IWYU pragma: keep
#include "nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/status_casters.h"
#include "xla/python/nb_helpers.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace nb = nanobind;

namespace nanobind {
namespace detail {

template <>
struct type_caster<xla::OpMetadata> {
 public:
  NB_TYPE_CASTER_FROM_PYTHON_ONLY(xla::OpMetadata,
                                  const_name("xla::OpMetadata"));

  bool from_python(handle h, uint8_t, cleanup_list*) noexcept {
    handle op_type = getattr(h, "op_type");
    if (!op_type.is_none()) {
      value.set_op_type(cast<std::string>(op_type));
    }
    handle op_name = getattr(h, "op_name");
    if (!op_name.is_none()) {
      value.set_op_name(cast<std::string>(op_name));
    }
    handle source_file = getattr(h, "source_file");
    if (!source_file.is_none()) {
      value.set_source_file(cast<std::string>(source_file));
    }
    handle source_line = getattr(h, "source_line");
    if (!source_line.is_none()) {
      value.set_source_line(cast<int32_t>(source_line));
    }
    return true;
  }
};

}  // namespace detail
}  // namespace nanobind

namespace xla {

namespace {

struct Uniquer {
  absl::Mutex mu;
  NameUniquer name_uniquer ABSL_GUARDED_BY(mu);
};

Uniquer* GetUniquer() {
  static Uniquer* uniquer = new Uniquer;
  return uniquer;
}

static std::string UniquifyName(const std::string& name) {
  Uniquer* uniquer = GetUniquer();
  absl::MutexLock lock(&uniquer->mu);
  return uniquer->name_uniquer.GetUniqueName(name);
}

}  // namespace

NB_MODULE(_xla_builder, m) {
  nb::class_<FrontendAttributes> frontend_attributes(m, "FrontendAttributes");
  frontend_attributes.def(nb::init<>())
      .def("__setitem__",
           [](FrontendAttributes* attr, std::string key, std::string value) {
             (*attr->mutable_map())[key] = value;
           });

  nb::class_<XlaOp> xla_op_class(m, "XlaOp");

  nb::class_<XlaBuilder>(m, "XlaBuilder")
      .def("__init__",
           [](XlaBuilder* self, const std::string& name) {
             new (self) XlaBuilder(UniquifyName(name));
           })
      // TODO(phawkins): delete capitalized names after updating callers.
      .def("Build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           nb::arg("root") = std::nullopt)
      .def("GetShape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def("build",
           xla::ValueOrThrowWrapper(
               [](XlaBuilder& builder, std::optional<XlaOp> root) {
                 return root ? builder.Build(*root) : builder.Build();
               }),
           "Builds a computation from the contents of the builder.",
           nb::arg("root") = std::nullopt)
      .def("clear_op_metadata", &XlaBuilder::ClearOpMetadata)
      .def("get_shape", xla::ValueOrThrowWrapper(&XlaBuilder::GetShape))
      .def(
          "get_program_shape",
          [](const XlaBuilder& builder,
             std::optional<XlaOp> root) -> ProgramShape {
            return ValueOrThrow(root ? builder.GetProgramShape(*root)
                                     : builder.GetProgramShape());
          },
          nb::arg("root") = std::nullopt)
      .def("is_constant", xla::ValueOrThrowWrapper(&XlaBuilder::IsConstant))
      .def("set_op_metadata", &XlaBuilder::SetOpMetadata)
      .def("set_sharding", &XlaBuilder::SetSharding)
      .def("clear_sharding", &XlaBuilder::ClearSharding)
      .def("set_frontend_attributes", &XlaBuilder::SetFrontendAttributes)
      .def("clear_frontend_attributes", &XlaBuilder::ClearFrontendAttributes)
      .def("setup_alias",
           [](XlaBuilder& builder, const std::vector<int64_t>& output_index,
              int64_t param_number, const std::vector<int64_t>& param_index) {
             builder.SetUpAlias(
                 ShapeIndex(output_index.begin(), output_index.end()),
                 param_number,
                 ShapeIndex(param_index.begin(), param_index.end()));
           });
}

}  // namespace xla
