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

#include "xla/python/py_client_cpu.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "xla/ffi/api/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/python/callback.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/py_host_callback.h"
#include "xla/python/types.h"
#include "xla/shape_util.h"

namespace nb = nanobind;

namespace xla {

ffi::Error XlaFfiPythonCpuCallback(ifrt::FfiLoadedHostCallbacks* callbacks,
                                   uint64_t index, ffi::RemainingArgs args,
                                   ffi::RemainingRets rets) {
  auto loaded_callback = static_cast<PyCpuLoadedHostCallback*>(
      callbacks->callbacks->at(index).get());
  CpuCallback* callback = loaded_callback->cpu_callback();

  nb::gil_scoped_acquire gil;
  auto nb_args = nb::steal<nb::tuple>(PyTuple_New(args.size()));
  for (size_t i = 0; i < args.size(); ++i) {
    auto arg = args.get<ffi::AnyBuffer>(i);
    auto ptype = static_cast<PrimitiveType>(arg->element_type());
    if (ptype == TOKEN) {
      PyTuple_SET_ITEM(nb_args.ptr(), i, nb::none().release().ptr());
      continue;
    }
    auto maybe_dtype = PrimitiveTypeToNbDtype(ptype);
    if (!maybe_dtype.ok()) {
      return ffi::Error::Internal(maybe_dtype.status().ToString());
    }
    auto dtype = maybe_dtype.value();
    auto dims = absl::Span<const int64_t>(arg->dimensions().begin(),
                                          arg->dimensions().size());
    // We pass in data using default numpy layout i.e., std::nullopt.
    auto array =
        nb_numpy_ndarray(dtype, dims, std::nullopt, arg.value().untyped_data());
    array.attr("flags").attr("writeable") = nb::bool_(false);
    PyTuple_SET_ITEM(nb_args.ptr(), i, array.release().ptr());
  }

  EnterHostCallback();
  // TODO(dsuo): Change this to use the Python vectorcall protocol, which allows
  // you to avoid constructing a tuple for the arguments.
  absl::StatusOr<nb::tuple> maybe_result_tuple =
      callback->FfiCall(std::move(nb_args));
  LeaveHostCallback();
  if (!maybe_result_tuple.ok()) {
    return ffi::Error::Internal(maybe_result_tuple.status().ToString());
  }
  auto result_tuple = maybe_result_tuple.value();

  for (size_t i = 0; i < rets.size(); ++i) {
    auto ret = rets.get<ffi::AnyBuffer>(i).value();
    auto ptype = static_cast<PrimitiveType>(ret->element_type());
    if (ptype == TOKEN) continue;
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    nb_numpy_ndarray array = nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    // We expect the output to be in default numpy layout.
    auto dims = absl::Span<const int64_t>(ret->dimensions().begin(),
                                          ret->dimensions().size());
    auto maybe_expected_shape = ShapeUtil::MakeValidatedShape(ptype, dims);
    if (!maybe_expected_shape.ok()) {
      return ffi::Error::Internal(maybe_expected_shape.status().ToString());
    }
    auto expected_shape = maybe_expected_shape.value();
    auto expected_strides = ByteStridesForShape(expected_shape);
    if (strides == expected_strides) {
      std::memcpy(ret->untyped_data(), array.data(), ret->size_bytes());
      continue;
    }
    xla::TransposePlan::Options options;
    options.elem_size_in_bytes = xla::primitive_util::ByteWidth(ptype);
    options.dims = absl::Span<const int64_t>(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::InlinedVector<int64_t, 4> reversed_layout;
    reversed_layout.resize(expected_shape.dimensions_size());
    absl::c_reverse_copy(expected_shape.layout().minor_to_major(),
                         reversed_layout.begin());
    options.permutation = reversed_layout;
    options.input_layout = xla::TransposePlan::Striding{strides};
    auto maybe_plan = callback->transpose_cache().GetOrCreate(options);
    if (!maybe_plan.ok()) {
      return ffi::Error::Internal(maybe_plan.status().ToString());
    }
    auto plan = maybe_plan.value();
    plan->Execute(array.data(), ret->untyped_data());
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    kXlaFfiPythonCpuCallback, XlaFfiPythonCpuCallback,
    ffi::Ffi::Bind()
        .Ctx<ffi::UserData<ifrt::FfiLoadedHostCallbacks>>()
        .Attr<uint64_t>("index")
        .RemainingArgs()
        .RemainingRets());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla_ffi_python_cpu_callback",
                         "HOST", kXlaFfiPythonCpuCallback);

}  // namespace xla
