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

#include "absl/strings/str_cat.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/shared_ptr.h"  // IWYU pragma: keep
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/numpy.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
// The "third_party/pybind11_abseil/status_casters.h" header says
// it's deprecated and that we should import the other headers directly.
#include "pybind11_abseil/import_status_module.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "xla/literal.h"
#include "xla/pjrt/status_casters.h"
#include "xla/python/logging.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/types.h"
#include "xla/xla_data.pb.h"
// NOTE: The tsl-numpy header forbids importing the actual NumPy arrayobject.h
// header before tsl-numpy (whereas, importing pybind11-numpy before tsl-numpy
// is fine); however, tsl-numpy does reexport NumPy's arrayobject.h header.
// Since one of the TF headers above already includes tsl-numpy, therefore
// we must include it down here rather than including actual NumPy directly.
#include "xla/tsl/python/lib/core/numpy.h"

namespace py = ::pybind11;
namespace nb = ::nanobind;

namespace {
py::object MakeNdarray(const xla::LiteralProto& proto) {
  auto m_lit = xla::Literal::CreateFromProto(proto);
  if (!m_lit.ok()) {
    // NOTE: The OSS version of XLA is still using an old version of
    // Abseil (LTS branch, Aug 2023, Patch 1) which does not have the
    // `AbslStringify` interface for implicitly converting `absl::Status`
    // into the `absl::AlphaNum` required by `absl::StrCat`.  Therefore we
    // inline the latest definition of the `AbslStringify` overload.
    throw py::value_error(absl::StrCat(
        "Cannot `xla::Literal::CreateFromProto`: ",
        m_lit.status().ToString(absl::StatusToStringMode::kWithEverything)));
  }

  // Move (not copy) the literal onto the heap, for sharing with Python.
  auto lit = std::make_shared<xla::Literal>(std::move(m_lit).value());

  auto nbobj = xla::ValueOrThrow(xla::LiteralToPython(std::move(lit)));
  return py::reinterpret_steal<py::object>(nbobj.release().ptr());
}

// Partial reversion of cl/617156835, until we can get the proto-casters
// (and hence the extension) switched over to nanobind.
// TODO(wrengr): Or can we mix `{py,nb}::module_::def` calls??
xla::PrimitiveType DtypeToEtype(const py::dtype& py_d) {
  auto nb_d = nb::borrow<xla::nb_dtype>(py_d.ptr());
  return xla::ValueOrThrow(xla::DtypeToPrimitiveType(nb_d));
}

py::dtype EtypeToDtype(xla::PrimitiveType p) {
  auto nb_d = xla::ValueOrThrow(xla::PrimitiveTypeToNbDtype(p));
  return py::reinterpret_steal<py::dtype>(nb_d.release().ptr());
}
}  // namespace

// NOTE: It seems insurmountable to get "native_proto_caster.h" to work
// with nanobind modules; therefore, we define our extension as a pybind11
// module so that we can use `pybind11::module_::def`.
PYBIND11_MODULE(_types, py_m) {
  // Initialize ABSL logging because code within XLA uses it.
  // (As per `xla::Init` in "xla.cc"; though we don't need it ourselves.)
#ifndef PLATFORM_GOOGLE
  xla::InitializeAbslLogging();
#endif  // PLATFORM_GOOGLE

  // Normally this would happen at the start of NB_MODULE, but since
  // this is a pybind11 module we have to do this ourselves.
  // (As per `xla::Init` in "xla.cc".)
  nb::detail::init(NB_DOMAIN_STR);

  // Import implicit conversions from Python protobuf objects to C++
  // protobuf objects.
  pybind11_protobuf::ImportNativeProtoCasters();

  // Import dependencies for converting `absl::StatusOr` to Python exceptions.
  // This also brings into scope pybind11 casters for doing conversions
  // implicitly; however, towards the goal of converting everything to
  // nanobind, we call `xla::ValueOrThrow` to make make the conversions
  // explicit (since `nb::detail::type_caster` disallows raising exceptions,
  // and therefore nanobind cannot do this implicitly).
  py::google::ImportStatusModule();

  // Import the 'ml_dtypes' module; which is implicitly required by
  // `xla::LiteralToPython`.
  // NOTE: If the `tsl_pybind_extension` build rule allowed us to specify
  // this as a py_dep, then importing the module here would mean that
  // client Python code need not import the hidden dependency themselves.
  // However, since `tsl_pybind_extension` does not allow specifying py_deps,
  // if client rules do not themselves declare the dependency then this will
  // generate a `ModuleNotFoundError` / `ImportError` exception.  Hence why
  // we define the "types.py" wrapper library to encapsulate the dependency.
  py::module_::import("ml_dtypes");

  // Ensure that tsl-numpy initializes datastructures of the actual-NumPy
  // implementation, and does whatever else tsl-numpy needs.  This is
  // also necessary for using the `xla::nb_dtype` type.
  tsl::ImportNumpy();

  // Declare that C++ can `nb::cast` from `std::shared_ptr<xla::Literal>`
  // to `nb::object`; which is implicitly required by `xla::LiteralToPython`.
  // (FWIW: This also enables using `nb::type<xla::Literal>()` to get
  // the Python-type-object associated with the C++ class.)
  //
  // NOTE: This does *not* mean that C++ can `py::cast` from `xla::Literal`
  // to `py::object`.  It's unclear whether we can simultaneously provide
  // both nanobind and pybind11 bindings (if we wanted the latter).
  nb::module_ nb_m = nb::cast<nb::module_>(nb::borrow(py_m.ptr()));
  nb::class_<xla::Literal>(nb_m, "Literal")
      .def("__repr__", &xla::Literal::ToString);

  // We do not define `py_m.doc()` here, since it wouldn't be inherited
  // by the "types.py" wrapper library.  See there for the python docstring.

  // LINT.IfChange
  py_m.def("make_ndarray", &MakeNdarray, py::arg("proto").none(false),
           py::pos_only(), R"pbdoc(
    Converts `tensorflow.compiler.xla.xla_data_pb2.LiteralProto`
    into an `xla::Literal` and then converts that literal into a tree
    of tuples with leaves being `numpy.ndarray` views of array-shaped
    sub-literals.
  )pbdoc");

  // This method name is based on `xla_client.dtype_to_etype`.
  // NOTE: `xla_client` uses a Python class wrapping the protobuf-enum,
  // rather than using the protobuf-enum directly.  See the module docstring
  // in "types.py" for more explanation on why.
  py_m.def("dtype_to_etype", &DtypeToEtype, py::arg("dtype").none(false),
           py::pos_only(), R"pbdoc(
    Converts `numpy.dtype` into
    `tensorflow.compiler.xla.xla_data_pb2.PrimitiveType`.
  )pbdoc");

  py_m.def("etype_to_dtype", &EtypeToDtype, py::arg("ptype").none(false),
           py::pos_only(), R"pbdoc(
    Converts `tensorflow.compiler.xla.xla_data_pb2.PrimitiveType` into
    `numpy.dtype`.
  )pbdoc");
  // LINT.ThenChange(_types.pyi)
}
