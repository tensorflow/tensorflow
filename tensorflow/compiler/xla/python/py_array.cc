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

#include "tensorflow/compiler/xla/python/py_array.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/status_casters.h"

namespace xla {
namespace {

namespace py = pybind11;

std::vector<std::shared_ptr<PjRtBuffer>> CreatePjRtBuffersFromPyBuffers(
    absl::Span<const PyBuffer::object> py_buffers) {
  std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers;
  pjrt_buffers.reserve(py_buffers.size());

  for (const auto& py_buffer : py_buffers) {
    pjrt_buffers.push_back(py_buffer.buf()->shared_ptr_buffer());
  }

  return pjrt_buffers;
}

std::vector<std::shared_ptr<PjRtBuffer>>
CreatePjRtBuffersFromSingleShardedPyArrays(
    absl::Span<const PyArray* const> py_arrays) {
  std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers;
  pjrt_buffers.reserve(py_arrays.size());

  for (const auto* py_array : py_arrays) {
    DCHECK_EQ(py_array->num_shards(), 1);
    pjrt_buffers.push_back(py_array->GetSharedPtrBuffer(0));
  }

  return pjrt_buffers;
}

}  // namespace

PyArray::PyArray(py::object aval, py::object sharding,
                 const std::vector<const PyArray*>& py_arrays, bool committed,
                 bool skip_checks, py::object fast_path_args)
    : PyArray(aval, pybind11::cast<bool>(aval.attr("weak_type")),
              DtypeToPrimitiveType(aval.attr("dtype")).ValueOrDie(),
              pybind11::cast<std::vector<int64_t>>(aval.attr("shape")),
              std::move(sharding), py_arrays.at(0)->py_client(),
              Traceback::Get(),
              CreatePjRtBuffersFromSingleShardedPyArrays(py_arrays), committed,
              skip_checks, std::move(fast_path_args)) {}

PyArray::PyArray(py::object aval, py::object sharding,
                 absl::Span<const PyBuffer::object> py_buffers, bool committed,
                 bool skip_checks, py::object fast_path_args)
    : PyArray(aval, pybind11::cast<bool>(aval.attr("weak_type")),
              DtypeToPrimitiveType(aval.attr("dtype")).ValueOrDie(),
              pybind11::cast<std::vector<int64_t>>(aval.attr("shape")),
              std::move(sharding), py_buffers.at(0).buf()->client(),
              Traceback::Get(), CreatePjRtBuffersFromPyBuffers(py_buffers),
              committed, skip_checks, std::move(fast_path_args)) {}

PyArray::PyArray(py::object aval, bool weak_type, PrimitiveType dtype,
                 std::vector<int64_t> shape, py::object sharding,
                 std::shared_ptr<PyClient> py_client,
                 std::shared_ptr<Traceback> traceback,
                 std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers,
                 bool committed, bool skip_checks, py::object fast_path_args)
    : aval_(std::move(aval)),
      weak_type_(weak_type),
      dtype_(dtype),
      shape_(std::move(shape)),
      sharding_(std::move(sharding)),
      fast_path_args_(std::move(fast_path_args)),
      committed_(std::move(committed)),
      py_client_(std::move(py_client)),
      traceback_(std::move(traceback)),
      pjrt_buffers_(std::move(pjrt_buffers)) {
  if (!skip_checks) {
    Check();
    Rearrange();
  }
}

void PyArray::Check() {
  try {
    py::cast(this).attr("_check")();
  } catch (py::error_already_set& err) {
    throw py::value_error(err.what());
  }
}

void PyArray::Rearrange() { py::cast(this).attr("_rearrange")(); }

py::object PyArray::arrays() const {
  if (pjrt_buffers_.empty()) return py::none();

  std::vector<PyBuffer::object> py_buffers;
  py_buffers.reserve(pjrt_buffers_.size());
  for (const auto& pjrt_buffer : pjrt_buffers_) {
    py_buffers.push_back(PyBuffer::Make(py_client_, pjrt_buffer, traceback_));
  }

  return py::cast(py_buffers);
}

Status PyArray::set_arrays(py::object obj) {
  if (obj.is_none()) {
    pjrt_buffers_.clear();
    return Status::OK();
  }

  if (!py::isinstance<py::list>(obj)) {
    return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                           py::cast<std::string>(py::str(obj.get_type())));
  }

  py::list list = obj;

  if (list.empty()) return Status::OK();

  pjrt_buffers_.clear();
  pjrt_buffers_.reserve(list.size());
  for (py::handle obj : list) {
    // TODO(chky): Currently only List[Buffer] is handled here. We need to
    // handle List[Array] as well.
    if (obj.get_type().ptr() != PyBuffer::type()) {
      return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                             py::cast<std::string>(py::str(obj.get_type())));
    }

    auto* py_buffer = PyBuffer::AsPyBufferUnchecked(obj);
    DCHECK_EQ(py_buffer->client(), py_client_);
    pjrt_buffers_.push_back(py_buffer->shared_ptr_buffer());
  }
  return Status::OK();
}

void PyArray::RegisterTypes(py::module& m) {
  py::class_<PyArray>(m, "Array", py::dynamic_attr())
      .def(py::init<py::object, py::object, absl::Span<const PyBuffer::object>,
                    bool, bool, py::object>(),
           py::arg("aval"), py::arg("sharding"), py::arg("arrays"),
           py::arg("committed"), py::arg("_skip_checks") = false,
           py::arg("_fast_path_args") = py::none())
      .def(py::init<py::object, py::object, const std::vector<const PyArray*>&,
                    bool, bool, py::object>(),
           py::arg("aval"), py::arg("sharding"), py::arg("arrays"),
           py::arg("committed"), py::arg("_skip_checks") = false,
           py::arg("_fast_path_args") = py::none())
      .def_property_readonly("_sharding", &PyArray::sharding)
      .def_property("aval", &PyArray::aval, &PyArray::set_aval)
      .def_property("_arrays", &PyArray::arrays, &PyArray::set_arrays)
      .def_property_readonly("_fast_path_args", &PyArray::fast_path_args)
      .def_property("_npy_value", &PyArray::npy_value, &PyArray::set_npy_value)
      .def_property_readonly("_committed", &PyArray::committed)
      .def("block_until_ready", [](py::object self) -> StatusOr<py::object> {
        TF_RETURN_IF_ERROR(py::cast<PyArray*>(self)->BlockUntilReady());
        return self;
      });
}

}  // namespace xla
