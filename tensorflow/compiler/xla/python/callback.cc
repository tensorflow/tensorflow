/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/callback.h"

#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/tsl/profiler/lib/traceme.h"

namespace py = pybind11;

namespace xla {

Status CpuCallback::PrepareAndCallInternal(void* result, void** arg_ptrs) {
  absl::Span<void* const> inputs(arg_ptrs, args_.size());
  absl::Span<void* const> outputs(reinterpret_cast<void**>(result),
                                  results_.size());

  py::gil_scoped_acquire gil;
  py::tuple args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (args_[i].type == xla::TOKEN) {
      args[i] = py::none();
    } else {
      args[i] = py::array(args_[i].dtype, args_[i].dims, args_[i].strides,
                          const_cast<void*>(inputs[i]));
      args[i].attr("flags").attr("writeable") = Py_False;
    }
  }

  TF_ASSIGN_OR_RETURN(auto result_tuple, CallInternal(std::move(args)));

  for (size_t i = 0; i < results_.size(); ++i) {
    py::object output = py::reinterpret_borrow<py::object>(
        PyTuple_GetItem(result_tuple.ptr(), i));
    py::array array = py::cast<py::array>(std::move(output));
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    absl::Span<int64_t const> strides(
        reinterpret_cast<const int64_t*>(array.strides()), array.ndim());
    if (strides == results_[i].expected_strides) {
      std::memcpy(outputs[i], array.data(), results_[i].size_in_bytes);
    } else {
      xla::StatusOr<std::shared_ptr<xla::TransposePlan>> plan =
          transpose_cache_.GetOrCreate(
              xla::primitive_util::ByteWidth(results_[i].type), dims,
              results_[i].reversed_layout,
              /*input_layout=*/xla::TransposePlan::Striding{strides});
      if (!plan.ok()) {
        return std::move(plan).status();
      }
      plan.value()->Execute(array.data(), outputs[i]);
    }
  }

  return OkStatus();
}

void CpuCallback::PrepareAndCall(void* result, void** arg_ptrs,
                                 XlaCustomCallStatus* status) {
  auto s = PrepareAndCallInternal(result, arg_ptrs);
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
    return;
  }
}

Status CpuCallback::PrepareAndCall(void* result, void** arg_ptrs) {
  return PrepareAndCallInternal(result, arg_ptrs);
}

StatusOr<py::tuple> CpuCallback::CallInternal(py::tuple args) {
  py::object result_object;
  try {
    result_object = callable_(*py::reinterpret_borrow<py::args>(args));
  } catch (py::error_already_set& e) {
    PyErr_Clear();
    std::string error_message = e.what();
    return InternalError("CpuCallback error: %s", error_message);
  }
  if (!PyTuple_Check(result_object.ptr())) {
    return InternalError("CPU callback expected a tuple result, got %s",
                         static_cast<std::string>(py::repr(result_object)));
  }
  if (PyTuple_Size(result_object.ptr()) != results_.size()) {
    return InternalError(
        "CPU callback expected a tuple with %d results, got %d",
        results_.size(), PyTuple_Size(result_object.ptr()));
  }
  py::tuple result_tuple = py::cast<py::tuple>(result_object);
  for (size_t i = 0; i < results_.size(); ++i) {
    py::object output = py::reinterpret_borrow<py::object>(
        PyTuple_GetItem(result_tuple.ptr(), i));
    if (results_[i].type == xla::TOKEN) {
      if (!output.is_none()) {
        return InternalError(
            "Token output from Python callback should be None, got %s",
            static_cast<std::string>(py::repr(output)));
      }
      continue;
    }
    py::array array = py::cast<py::array>(std::move(output));
    static_assert(sizeof(ssize_t) == sizeof(int64_t),
                  "Expected ssize_t to be of equal size to int64_t");
    absl::Span<int64_t const> dims(
        reinterpret_cast<const int64_t*>(array.shape()), array.ndim());
    if (dims != results_[i].expected_dims) {
      return InternalError(
          "Mismatched result shape for %d-th return value from CPU callback; "
          "expected array with dimensions %s, got %s",
          i, absl::StrJoin(results_[i].expected_dims, ","),
          absl::StrJoin(dims, ","));
    }
  }
  return result_tuple;
}

StatusOr<py::tuple> CpuCallback::Call(py::tuple args) {
  return CallInternal(std::move(args));
}

std::optional<py::tuple> CpuCallback::Call(py::tuple args,
                                           XlaCustomCallStatus* status) {
  auto statusor = CallInternal(std::move(args));
  if (!statusor.ok()) {
    absl::string_view msg = statusor.status().message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
    return std::nullopt;
  }
  return std::move(statusor).value();
}

void XlaPythonCpuCallback(void* output, void** inputs,
                          XlaCustomCallStatus* status) {
  CpuCallback* callback =
      absl::bit_cast<CpuCallback*>(*static_cast<uintptr_t*>(inputs[0]));
  callback->PrepareAndCall(output, inputs + 1, status);
}

}  // namespace xla
