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
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "tensorflow/compiler/xla/python/python_utils.h"
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
    absl::Span<const PyArray> py_arrays) {
  std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers;
  pjrt_buffers.reserve(py_arrays.size());

  for (const auto& py_array : py_arrays) {
    DCHECK_EQ(py_array.num_shards(), 1);
    pjrt_buffers.push_back(py_array.GetSharedPtrBuffer(0));
  }

  return pjrt_buffers;
}

struct PyArrayObject {
  PyObject_HEAD;
  PyObject* weakrefs;
  alignas(PyArray::Storage) char array_storage[sizeof(PyArray::Storage)];
};
static_assert(std::is_standard_layout<PyArrayObject>::value);

PyArray::Storage* GetPyArrayStorageFromObject(PyArrayObject* py_array_object) {
  return std::launder(
      reinterpret_cast<PyArray::Storage*>(py_array_object->array_storage));
}

extern "C" PyObject* PyArray_tp_new(PyTypeObject* type, PyObject*, PyObject*) {
  PyObject* self = type->tp_alloc(type, 0);
  return self;
}

extern "C" void PyArray_tp_dealloc(PyObject* self) {
  PyTypeObject* tp = Py_TYPE(self);
  auto* obj = reinterpret_cast<PyArrayObject*>(self);

  if (obj->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }

  GetPyArrayStorageFromObject(obj)->~Storage();

  tp->tp_free(self);
  Py_DECREF(tp);
}

// dynamic_attr: Allow the garbage collector to traverse the internal instance
// `__dict__`.
extern "C" int PyArray_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_VISIT(dict);
// https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_traverse
#if PY_VERSION_HEX >= 0x03090000
  Py_VISIT(Py_TYPE(self));
#endif
  return 0;
}

// dynamic_attr: Allow the GC to clear the dictionary.
extern "C" int PyArray_tp_clear(PyObject* self) {
  PyObject*& dict = *_PyObject_GetDictPtr(self);
  Py_CLEAR(dict);
  return 0;
}

// Give instances of this type a `__dict__` and opt into garbage collection.
void EnableDynamicAttribute(PyHeapTypeObject* heap_type) {
  auto* type = &heap_type->ht_type;
  type->tp_flags |= Py_TPFLAGS_HAVE_GC;
#if PY_VERSION_HEX < 0x030B0000
  type->tp_dictoffset = type->tp_basicsize;  // place dict at the end
  type->tp_basicsize +=
      (ssize_t)sizeof(PyObject*);  // and allocate enough space for it
#else
  type->tp_flags |= Py_TPFLAGS_MANAGED_DICT;
#endif
  type->tp_traverse = PyArray_tp_traverse;
  type->tp_clear = PyArray_tp_clear;

  static PyGetSetDef getset[] = {{"__dict__", PyObject_GenericGetDict,
                                  PyObject_GenericSetDict, nullptr, nullptr},
                                 {nullptr, nullptr, nullptr, nullptr, nullptr}};
  type->tp_getset = getset;
}

template <typename... Args>
PyArray::Storage* Construct(PyArrayObject* self, Args&&... args) {
  return new (self->array_storage)
      PyArray::Storage(std::forward<Args>(args)...);
}

}  // namespace

void PyArray::PyInit(py::object self, py::object aval, py::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks) {
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            pybind11::cast<bool>(aval.attr("weak_type")), aval.attr("dtype"),
            pybind11::cast<std::vector<int64_t>>(aval.attr("shape")),
            std::move(sharding), committed, py_arrays.at(0).py_client(),
            Traceback::Get(),
            CreatePjRtBuffersFromSingleShardedPyArrays(py_arrays));

  PyArray py_array = self;

  if (!skip_checks) {
    py_array.CheckAndRearrange();
  }
}

void PyArray::PyInit(py::object self, py::object aval, py::object sharding,
                     absl::Span<const PyBuffer::object> py_buffers,
                     bool committed, bool skip_checks) {
  Construct(reinterpret_cast<PyArrayObject*>(self.ptr()), aval,
            pybind11::cast<bool>(aval.attr("weak_type")), aval.attr("dtype"),
            pybind11::cast<std::vector<int64_t>>(aval.attr("shape")),
            std::move(sharding), committed, py_buffers.at(0).buf()->client(),
            Traceback::Get(), CreatePjRtBuffersFromPyBuffers(py_buffers));

  PyArray py_array = self;

  if (!skip_checks) {
    py_array.CheckAndRearrange();
  }
}

PyArray::PyArray(py::object aval, bool weak_type, py::dtype dtype,
                 std::vector<int64_t> shape, py::object sharding,
                 std::shared_ptr<PyClient> py_client,
                 std::shared_ptr<Traceback> traceback,
                 std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers,
                 bool committed, bool skip_checks) {
  auto* self =
      PyArray_tp_new(reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr);
  ptr() = self;
  Construct(reinterpret_cast<PyArrayObject*>(self), std::move(aval), weak_type,
            std::move(dtype), std::move(shape), std::move(sharding), committed,
            std::move(py_client), std::move(traceback),
            std::move(pjrt_buffers));

  if (!skip_checks) {
    CheckAndRearrange();
  }
}

PyArray::Storage& PyArray::GetStorage() {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

const PyArray::Storage& PyArray::GetStorage() const {
  return *GetPyArrayStorageFromObject(reinterpret_cast<PyArrayObject*>(ptr()));
}

void PyArray::CheckAndRearrange() { this->attr("_check_and_rearrange")(); }

py::object PyArray::arrays() {
  // For performance, we only keep pjrt buffers by default. But on python side
  // "_arrays" returns PyBuffers instead, and subsequent calls to "_arrays"
  // should return the same PyBuffers (to avoid duplicate device to host
  // transfers). So we create PyBuffers the first time it is called and reuse
  // them later.
  if (pjrt_buffers().empty()) return py::none();

  auto& py_buffers = this->py_buffers();

  if (py_buffers.empty()) {
    py_buffers.reserve(pjrt_buffers().size());
    for (const auto& pjrt_buffer : pjrt_buffers()) {
      py_buffers.push_back(
          PyBuffer::Make(py_client(), pjrt_buffer, traceback()));
    }
  }

  return py::cast(py_buffers);
}

Status PyArray::set_arrays(py::object obj) {
  if (obj.is_none()) {
    pjrt_buffers().clear();
    py_buffers().clear();
    return OkStatus();
  }

  if (!py::isinstance<py::list>(obj)) {
    return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                           py::cast<std::string>(py::str(obj.get_type())));
  }

  py::list list = obj;

  if (list.empty()) return OkStatus();

  pjrt_buffers().clear();
  py_buffers().clear();
  pjrt_buffers().reserve(list.size());
  for (py::handle obj : list) {
    // TODO(chky): Currently only List[Buffer] is handled here. We need to
    // handle List[Array] as well.
    if (obj.get_type().ptr() != PyBuffer::type()) {
      return InvalidArgument("Unsupported arg when setting Array._arrays: %s",
                             py::cast<std::string>(py::str(obj.get_type())));
    }

    auto* py_buffer = PyBuffer::AsPyBufferUnchecked(obj);
    DCHECK_EQ(py_buffer->client(), py_client());
    pjrt_buffers().push_back(py_buffer->shared_ptr_buffer());
  }
  return OkStatus();
}

Status PyArray::BlockUntilReady() const {
  pybind11::gil_scoped_release gil_release;
  Status status;
  for (const auto& pjrt_buffer : pjrt_buffers()) {
    // PjRtBuffer::BlockHostUntilReady() fix up the error message because some
    // clients rely on it.
    auto s = pjrt_buffer->BlockHostUntilReady();
    if (!s.ok()) status = std::move(s);
  }
  return status;
}

Status PyArray::SetUpType() {
  static constexpr char kName[] = "Array";

  py::str name = py::str(kName);
  py::str qualname = py::str(kName);

  auto* heap_type = reinterpret_cast<PyHeapTypeObject*>(
      PyType_Type.tp_alloc(&PyType_Type, 0));
  // Caution: we must not call any functions that might invoke the GC until
  // PyType_Ready() is called below. Otherwise the GC might see a
  // half-constructed type object.
  if (!heap_type) {
    return Internal("Unable to create heap type object");
  }
  heap_type->ht_name = name.release().ptr();
  heap_type->ht_qualname = qualname.release().ptr();
  PyTypeObject* type = &heap_type->ht_type;
  type->tp_name = kName;
  type->tp_basicsize = sizeof(PyArrayObject);
  type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
  type->tp_new = PyArray_tp_new;
  type->tp_dealloc = PyArray_tp_dealloc;

  // Supported protocols
  type->tp_as_number = &heap_type->as_number;
  type->tp_as_sequence = &heap_type->as_sequence;
  type->tp_as_mapping = &heap_type->as_mapping;

  // Allow dynamic attributes.
  EnableDynamicAttribute(heap_type);

  // Allow weak references to DeviceArray objects.
  type->tp_weaklistoffset = offsetof(PyArrayObject, weakrefs);

  TF_RET_CHECK(PyType_Ready(type) == 0);

  PyArray::type_ = reinterpret_cast<PyObject*>(type);

  return OkStatus();
}

Status PyArray::RegisterTypes(py::module& m) {
  TF_RETURN_IF_ERROR(PyArray::SetUpType());
  auto type = py::reinterpret_borrow<py::object>(type_);
  m.attr("ArrayImpl") = type;

  type.attr("__init__") = py::cpp_function(
      [](py::object self, py::object aval, py::object sharding, py::list arrays,
         bool committed, bool skip_checks) {
        if (arrays[0].get_type().is(PyArray::type())) {
          auto py_arrays = py::cast<std::vector<PyArray>>(arrays);
          PyArray::PyInit(self, std::move(aval), std::move(sharding), py_arrays,
                          committed, skip_checks);
        } else {
          auto py_buffers = py::cast<std::vector<PyBuffer::object>>(arrays);
          PyArray::PyInit(self, std::move(aval), std::move(sharding),
                          py_buffers, committed, skip_checks);
        }
      },
      py::is_method(type), py::arg("aval"), py::arg("sharding"),
      py::arg("arrays"), py::arg("committed"), py::arg("_skip_checks") = false);
  type.attr("_sharding") = jax::property_readonly(&PyArray::sharding);
  type.attr("aval") = jax::property(&PyArray::aval, &PyArray::set_aval);
  type.attr("_arrays") = jax::property(&PyArray::arrays, &PyArray::set_arrays);
  type.attr("_npy_value") =
      jax::property(&PyArray::npy_value, &PyArray::set_npy_value);
  type.attr("_committed") = jax::property_readonly(&PyArray::committed);
  type.attr("block_until_ready") = py::cpp_function(
      [](PyArray self) -> StatusOr<py::object> {
        TF_RETURN_IF_ERROR(self.BlockUntilReady());
        return self;
      },
      py::is_method(type));
  type.attr("__module__") = m.attr("__name__");

  return OkStatus();
}

}  // namespace xla
