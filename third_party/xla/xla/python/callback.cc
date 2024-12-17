/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/callback.h"

#include <Python.h>
#include <sys/types.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "xla/pjrt/transpose.h"
#include "xla/primitive_util.h"
#include "xla/python/nb_numpy.h"
#include "xla/python/python_ref_manager.h"
#include "xla/service/custom_call_status.h"
#include "tsl/platform/statusor.h"

namespace nb = nanobind;

namespace xla {

CpuCallback::~CpuCallback() {
  // The destructor may be called without GIL held. In that case, we defer it
  // to GlobalPyRefManager.
  std::vector<nb::object> objects;
  objects.push_back(std::move(callable_));
  for (auto& arg : args_) {
    objects.push_back(std::move(arg.dtype));
  }

  GlobalPyRefManager()->AddGarbage(absl::MakeSpan(objects));
}

absl::Status CpuCallback::PrepareAndCall(void* result, void** arg_ptrs) {
  absl::Span<void* const> inputs(arg_ptrs, args_.size());
  absl::Span<void* const> outputs(reinterpret_cast<void**>(result),
                                  results_.size());

  nb::gil_scoped_acquire gil;
  nb::tuple args = nb::steal<nb::tuple>(PyTuple_New(inputs.size()));
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (args_[i].type == xla::TOKEN) {
      PyTuple_SET_ITEM(args.ptr(), i, nb::none().release().ptr());
    } else {
      nb_numpy_ndarray array =
          nb_numpy_ndarray(args_[i].dtype, args_[i].dims, args_[i].strides,
                           const_cast<void*>(inputs[i]));
      array.attr("flags").attr("writeable") = nb::bool_(false);
      PyTuple_SET_ITEM(args.ptr(), i, array.release().ptr());
    }
  }

  TF_ASSIGN_OR_RETURN(auto result_tuple, Call(std::move(args)));

  for (size_t i = 0; i < results_.size(); ++i) {
    if (results_[i].type == xla::TOKEN) {
      continue;
    }
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    nb_numpy_ndarray array = nb_numpy_ndarray::ensure(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == results_[i].expected_strides) {
      std::memcpy(outputs[i], array.data(), results_[i].size_in_bytes);
    } else {
      xla::TransposePlan::Options options;
      options.elem_size_in_bytes =
          xla::primitive_util::ByteWidth(results_[i].type);
      options.dims = dims;
      options.permutation = results_[i].reversed_layout;
      options.input_layout = xla::TransposePlan::Striding{strides};
      absl::StatusOr<std::shared_ptr<xla::TransposePlan>> plan =
          transpose_cache_.GetOrCreate(options);
      if (!plan.ok()) {
        return std::move(plan).status();
      }
      plan.value()->Execute(array.data(), outputs[i]);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<nb::tuple> CpuCallback::Call(nb::tuple args) {
  auto py_error_to_status = [](nb::python_error& e) {
    std::string error_message = e.what();
    return absl::InternalError(
        absl::StrFormat("CpuCallback error: %s", error_message));
  };
  nb::object result_object;
  try {
    result_object = callable_(*nb::borrow<nb::args>(args));
  } catch (nb::python_error& e) {
    return py_error_to_status(e);
  }
  if (!PyTuple_Check(result_object.ptr())) {
    return absl::InternalError(
        absl::StrFormat("CPU callback expected a tuple result, got %s",
                        nb::cast<absl::string_view>(nb::repr(result_object))));
  }
  if (PyTuple_Size(result_object.ptr()) != results_.size()) {
    return absl::InternalError(
        absl::StrFormat("CPU callback expected a tuple with %d results, got %d",
                        results_.size(), PyTuple_Size(result_object.ptr())));
  }
  nb::tuple result_tuple = nb::cast<nb::tuple>(result_object);
  for (size_t i = 0; i < results_.size(); ++i) {
    nb::object output =
        nb::borrow<nb::object>(PyTuple_GetItem(result_tuple.ptr(), i));
    if (results_[i].type == xla::TOKEN) {
      if (!output.is_none()) {
        return absl::InternalError(absl::StrFormat(
            "Token output from Python callback should be None, got %s",
            nb::cast<absl::string_view>(nb::repr(output))));
      }
      continue;
    }
    nb_numpy_ndarray array;
    try {
      array = nb_numpy_ndarray::from_any(output, NPY_ARRAY_ENSUREARRAY);
    } catch (nb::python_error& e) {
      return py_error_to_status(e);
    }
    static_assert(sizeof(ssize_t) == sizeof(int64_t),
                  "Expected ssize_t to be of equal size to int64_t");
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    if (dims != results_[i].expected_dims) {
      return absl::InternalError(absl::StrFormat(
          "Mismatched result shape for %d-th return value from CPU callback; "
          "expected array with dimensions %s, got %s",
          i, absl::StrJoin(results_[i].expected_dims, ","),
          absl::StrJoin(dims, ",")));
    }
  }
  return result_tuple;
}

void XlaPythonCpuCallback(void* output, void** inputs,
                          XlaCustomCallStatus* status) {
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(*static_cast<uintptr_t*>(inputs[0]));
  auto s = callback->PrepareAndCall(output, inputs + 1);
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
  }
}

}  // namespace xla
