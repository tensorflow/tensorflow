/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/py_buffer.h"

#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace py = pybind11;

namespace {

// Representation of a DeviceArrayBase as a Python object. Since
// a DeviceArrayBase has no fields, this is just a PyObject.
struct PyBufferBasePyObject {
  PyObject_HEAD;
};
static_assert(std::is_standard_layout<PyBufferBasePyObject>::value,
              "PyBufferBasePyObject must be standard layout");

// Representation of a DeviceArray as a Python object.
struct PyBufferPyObject {
  PyBufferBasePyObject base;
  PyBuffer buffer;
  // Used by the Python interpreter to maintain a list of weak references to
  // this object.
  PyObject* weakrefs;
};
static_assert(std::is_standard_layout<PyBufferPyObject>::value,
              "PyBufferPyObject must be standard layout");

PyObject* PyBuffer_tp_new(PyTypeObject* subtype, PyObject* args,
                          PyObject* kwds) {
  PyBufferPyObject* self =
      reinterpret_cast<PyBufferPyObject*>(subtype->tp_alloc(subtype, 0));
  if (!self) return nullptr;
  self->weakrefs = nullptr;
  return reinterpret_cast<PyObject*>(self);
}

void PyBuffer_tp_dealloc(PyObject* self) {
  PyTypeObject* tp = Py_TYPE(self);
  PyBufferPyObject* o = reinterpret_cast<PyBufferPyObject*>(self);
  if (o->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  o->buffer.~PyBuffer();
  tp->tp_free(self);
  Py_DECREF(tp);
}

}  // namespace

/*static*/ PyBuffer::object PyBuffer::Make(
    std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
    std::shared_ptr<Traceback> traceback) {
  py::object obj = py::reinterpret_steal<py::object>(PyBuffer_tp_new(
      reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr));
  PyBufferPyObject* buf = reinterpret_cast<PyBufferPyObject*>(obj.ptr());
  new (&buf->buffer)
      PyBuffer(std::move(client), std::move(buffer), std::move(traceback));
  return py::reinterpret_borrow<PyBuffer::object>(obj);
}

bool PyBuffer::IsPyBuffer(py::handle handle) {
  return handle.get_type() == PyBuffer::type();
}

/*static*/ PyBuffer* PyBuffer::AsPyBufferUnchecked(pybind11::handle handle) {
  return &(reinterpret_cast<PyBufferPyObject*>(handle.ptr())->buffer);
}

/*static*/ StatusOr<PyBuffer*> PyBuffer::AsPyBuffer(pybind11::handle handle) {
  if (!IsPyBuffer(handle)) {
    return InvalidArgument("Expected a DeviceArray");
  }
  return AsPyBufferUnchecked(handle);
}

py::handle PyBuffer::AsHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyBufferPyObject, buffer));
}

PyBuffer::PyBuffer(std::shared_ptr<PyClient> client,
                   std::shared_ptr<PjRtBuffer> buffer,
                   std::shared_ptr<Traceback> traceback)
    : client_(std::move(client)),
      buffer_(std::move(buffer)),
      traceback_(std::move(traceback)) {
  CHECK(PyGILState_Check());
  next_ = client_->buffers_[buffer_->device()->id()];
  client_->buffers_[buffer_->device()->id()] = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyBuffer::~PyBuffer() {
  CHECK(PyGILState_Check());
  if (client_->buffers_[device()->id()] == this) {
    client_->buffers_[device()->id()] = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

StatusOr<int64_t> PyBuffer::size() {
  Shape max_buffer_shape = buffer()->on_device_shape();
  if (max_buffer_shape.is_dynamic()) {
    TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
    return ShapeUtil::ElementsIn(*dynamic_shape);
  }
  return ShapeUtil::ElementsIn(max_buffer_shape);
}

StatusOr<const Shape*> PyBuffer::xla_dynamic_shape() {
  CHECK(PyGILState_Check());
  if (buffer_->on_device_shape().is_static()) {
    return &buffer_->on_device_shape();
  }
  // Python buffer protocol references shape data by pointer, therefore we must
  // store a valid copy of the shape.
  if (!dynamic_shape_) {
    Shape dynamic_shape;
    {
      py::gil_scoped_release gil_release;
      TF_ASSIGN_OR_RETURN(dynamic_shape, buffer_->logical_on_device_shape());
    }
    dynamic_shape_ = dynamic_shape;
  }
  return &dynamic_shape_.value();
}

pybind11::tuple PyBuffer::python_shape() const {
  return SpanToTuple(buffer()->on_device_shape().dimensions());
}

pybind11::dtype PyBuffer::python_dtype() const {
  PrimitiveType primitive = buffer()->on_device_shape().element_type();
  return PrimitiveTypeToDtype(primitive).ValueOrDie();
}

ClientAndPtr<PjRtDevice> PyBuffer::device() const {
  return WrapWithClient(client_, buffer_->device());
}

PyBuffer::object PyBuffer::Clone() const {
  auto buffer = Make(client_, buffer_, traceback_);
  buffer.buf()->sticky_device_ = sticky_device_;
  buffer.buf()->aval_ = aval_;
  return buffer;
}

StatusOr<py::object> PyBuffer::CopyToDevice(
    const ClientAndPtr<PjRtDevice>& dst_device) const {
  CHECK(dst_device.get() != nullptr);
  auto transfer_guard_formatter = [this, &dst_device] {
    auto shape = py::cast<std::string>(py::str(python_shape()));
    auto dtype = py::cast<std::string>(py::str(python_dtype()));
    return absl::StrCat("shape=", shape, ", dtype=", dtype,
                        ", device=", device()->DebugString(),
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

  GlobalPyRefManager()->CollectGarbage();
  std::unique_ptr<PjRtBuffer> out;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(out, buffer_->CopyToDevice(dst_device.get()));
  }
  auto traceback = Traceback::Get();
  return Make(dst_device.client, std::move(out), std::move(traceback));
}

std::pair<Status, bool> PyBuffer::CopyToRemoteDevice(
    absl::string_view serialized_descriptor) const {
  absl::Mutex mu;
  bool done = false;
  Status status;
  bool sends_were_enqueued;
  buffer_->CopyToRemoteDevice(
      serialized_descriptor,
      [&done, &status, &sends_were_enqueued, &mu](Status s, bool dispatched) {
        absl::MutexLock l(&mu);
        done = true;
        status = s;
        sends_were_enqueued = dispatched;
      });
  {
    py::gil_scoped_release gil_release;
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(
        +[](bool* done) { return *done; }, &done));
  }
  return std::make_pair(status, sends_were_enqueued);
}

Status PyBuffer::BlockHostUntilReady() {
  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return buffer_->BlockHostUntilReady();
}

Status PyBuffer::CopyToHostAsync() {
  if (!buffer_->IsOnCpu() && !host_value_) {
    auto transfer_guard_formatter = [this] {
      auto shape = py::cast<std::string>(py::str(python_shape()));
      auto dtype = py::cast<std::string>(py::str(python_dtype()));
      return absl::StrCat("shape=", shape, ", dtype=", dtype,
                          ", device=", device()->DebugString());
    };
    TF_RETURN_IF_ERROR(
        jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

    std::shared_ptr<HostValue> host_value = std::make_shared<HostValue>();
    host_value_ = host_value;
    // TODO(b/182461453): This is a blocking call. If we further implemented
    // populating dynamic shape metadata while fetching the literal, we wouldn't
    // need this static approach.
    TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());

    py::gil_scoped_release gil;
    host_value->value = std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(*dynamic_shape));
    Literal* literal = host_value->value.get();
    buffer_->ToLiteral(literal,
                       [host_value{std::move(host_value)}](Status status) {
                         host_value->status = std::move(status);
                         host_value->ready.Notify();
                       });
  }
  return OkStatus();
}

StatusOr<pybind11::object> PyBuffer::AsNumPyArray(py::handle this_obj) {
  if (buffer_->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  TF_RET_CHECK(buffer_->on_device_shape().IsArray());
  // On CPU, we can return the value in a zero-copy way.
  if (buffer_->IsOnCpu()) {
    TF_ASSIGN_OR_RETURN(const auto* shape, xla_dynamic_shape());
    TF_ASSIGN_OR_RETURN(py::dtype dtype,
                        PrimitiveTypeToDtype(shape->element_type()));
    // Objects that must be kept alive while the array is alive.
    struct Hold {
      py::object buffer;
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
    };
    auto hold = std::make_unique<Hold>();
    TF_ASSIGN_OR_RETURN(hold->external_reference_hold,
                        buffer_->AcquireExternalReference());
    hold->buffer = py::reinterpret_borrow<py::object>(this_obj);
    void* data = hold->external_reference_hold->OpaqueDeviceMemoryDataPointer();
    py::capsule hold_capsule(hold.release(),
                             [](void* h) { delete static_cast<Hold*>(h); });
    py::array array(dtype, shape->dimensions(), ByteStridesForShape(*shape),
                    data, hold_capsule);
    array.attr("flags").attr("writeable") = Py_False;
    {
      py::gil_scoped_release gil;
      TF_RETURN_IF_ERROR(buffer_->BlockHostUntilReady());
    }
    return array;
  }

  TF_RETURN_IF_ERROR(CopyToHostAsync());
  if (!host_value_->ready.HasBeenNotified()) {
    py::gil_scoped_release gil;
    host_value_->ready.WaitForNotification();
  }
  TF_RETURN_IF_ERROR(host_value_->status);
  TF_ASSIGN_OR_RETURN(py::object array, LiteralToPython(host_value_->value));
  array.attr("flags").attr("writeable") = Py_False;
  return array;
}

StatusOr<std::uintptr_t> PyBuffer::UnsafeBufferPointer() const {
  return client_->pjrt_client()->UnsafeBufferPointer(buffer_.get());
}

StatusOr<py::dict> PyBuffer::CudaArrayInterface() {
  // TODO(zhangqiaorjc): Differentiate between NVidia and other GPUs.
  if (buffer_->client()->platform_id() != GpuId()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!buffer_->on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (buffer_->on_device_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      buffer_->on_device_shape().layout()));

  py::dict result;
  TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
  result["shape"] = SpanToTuple(dynamic_shape->dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          buffer_->on_device_shape().element_type()));
  result["typestr"] = std::move(typestr);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      buffer_->AcquireExternalReference());
  const void* root_ptr =
      external_reference_hold->OpaqueDeviceMemoryDataPointer();
  py::tuple data(2);
  data[0] = py::int_(absl::bit_cast<std::uintptr_t>(root_ptr));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

// PEP 3118 buffer protocol implementation.

namespace {

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold)
      : external_reference_hold(std::move(external_reference_hold)) {}

  std::string format;
  std::vector<Py_ssize_t> strides;
  // We keep an external reference hold to the PjRtBuffer. This prevents a
  // use-after-free in the event that Delete() is called on a buffer with an
  // live buffer protocol view. It does however mean that Delete() sometimes
  // won't actually delete immediately.
  std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
};

int PyBuffer_bf_getbuffer(PyObject* exporter, Py_buffer* view, int flags) {
  Status status = [&]() {
    TF_ASSIGN_OR_RETURN(PyBuffer * py_buffer, PyBuffer::AsPyBuffer(exporter));
    PjRtBuffer& buffer = *py_buffer->buffer();
    TF_ASSIGN_OR_RETURN(const auto* shape, py_buffer->xla_dynamic_shape());
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (!buffer.IsOnCpu()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }
    if (!buffer.on_device_shape().IsArray()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for array buffers.");
    }
    // If we allowed exports of formatted BF16 buffers, consumers would get
    // confused about the type because there is no way to describe BF16 to
    // Python.
    if (buffer.on_device_shape().element_type() == BF16 &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "bfloat16 buffer format not supported by Python buffer protocol.");
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
      return InvalidArgument("XLA buffers are read-only.");
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
        buffer.AcquireExternalReference());
    if (buffer.IsDeleted()) {
      return InvalidArgument("Deleted buffer used in buffer protocol.");
    }

    if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
         (flags & PyBUF_STRIDES) == PyBUF_ND) &&
        !LayoutUtil::IsMonotonicWithDim0Major(shape->layout())) {
      return InvalidArgument("Buffer is not in C-contiguous layout.");
    } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape->layout())) {
      return InvalidArgument("Buffer is not in F-contiguous layout.");
    } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Major(shape->layout()) &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape->layout())) {
      return InvalidArgument("Buffer is not in contiguous layout.");
    }
    std::memset(view, 0, sizeof(Py_buffer));
    const void* root_ptr =
        external_reference_hold->OpaqueDeviceMemoryDataPointer();
    view->buf = const_cast<void*>(root_ptr);
    auto extra =
        std::make_unique<ExtraBufferInfo>(std::move(external_reference_hold));
    view->itemsize = ShapeUtil::ByteSizeOfPrimitiveType(shape->element_type());
    view->len = ShapeUtil::ByteSizeOf(*shape);
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      TF_ASSIGN_OR_RETURN(extra->format, FormatDescriptorForPrimitiveType(
                                             shape->element_type()));
      view->format = const_cast<char*>(extra->format.c_str());
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = shape->dimensions_size();
      static_assert(sizeof(int64_t) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64_t*>(shape->dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides = ByteStridesForShape(*shape);
          view->strides = extra->strides.data();
        }
      }
    }
    TF_RETURN_IF_ERROR(buffer.BlockHostUntilReady());
    view->internal = extra.release();
    return OkStatus();
  }();
  if (!status.ok()) {
    // numpy.asarray(...) silents the PyExc_BufferError. Adding a log here helps
    // debugging when the error really occurs.
    VLOG(1) << "Buffer Protocol Error: " << status;
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PyBuffer_bf_releasebuffer(PyObject*, Py_buffer* buffer) {
  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

PyBufferProcs PyBuffer_tp_as_buffer = []() {
  PyBufferProcs procs;
  procs.bf_getbuffer = &PyBuffer_bf_getbuffer;
  procs.bf_releasebuffer = &PyBuffer_bf_releasebuffer;
  return procs;
}();

}  // namespace

PyObject* PyBuffer::base_type_ = nullptr;
PyObject* PyBuffer::type_ = nullptr;

Status PyBuffer::RegisterTypes(py::module& m) {
  // We do not use pybind11::class_ to build Python wrapper objects because
  // creation, destruction, and casting of buffer objects is performance
  // critical. By using hand-written Python classes, we can avoid extra C heap
  // allocations, and we can avoid pybind11's slow cast<>() implementation
  // during jit dispatch.

  // We need to use heap-allocated type objects because we want to add
  // additional methods dynamically.
  {
    py::str name = py::str("DeviceArrayBase");
    py::str qualname = py::str("DeviceArrayBase");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
        PyType_Type.tp_alloc(&PyType_Type, 0));
    // Caution: we must not call any functions that might invoke the GC until
    // PyType_Ready() is called. Otherwise the GC might see a half-constructed
    // type object.
    if (!heap_type) {
      return Internal("Unable to create heap type object");
    }
    heap_type->ht_name = name.release().ptr();
    heap_type->ht_qualname = qualname.release().ptr();
    PyTypeObject* type = &heap_type->ht_type;
    type->tp_name = "DeviceArrayBase";
    type->tp_basicsize = sizeof(PyBufferBasePyObject);
    type->tp_flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE | Py_TPFLAGS_BASETYPE;
    TF_RET_CHECK(PyType_Ready(type) == 0);
    base_type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object base_type = py::reinterpret_borrow<py::object>(base_type_);
  base_type.attr("__module__") = m.attr("__name__");
  m.attr("DeviceArrayBase") = base_type;

  {
    py::tuple bases = py::make_tuple(base_type);
    py::str name = py::str("DeviceArray");
    py::str qualname = py::str("DeviceArray");
    PyHeapTypeObject* heap_type = reinterpret_cast<PyHeapTypeObject*>(
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
    type->tp_name = "DeviceArray";
    type->tp_basicsize = sizeof(PyBufferPyObject);
    type->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HEAPTYPE;
    type->tp_bases = bases.release().ptr();
    type->tp_dealloc = PyBuffer_tp_dealloc;
    type->tp_new = PyBuffer_tp_new;
    // Supported protocols
    type->tp_as_number = &heap_type->as_number;
    type->tp_as_sequence = &heap_type->as_sequence;
    type->tp_as_mapping = &heap_type->as_mapping;
    type->tp_as_buffer = &PyBuffer_tp_as_buffer;

    // Allow weak references to DeviceArray objects.
    type->tp_weaklistoffset = offsetof(PyBufferPyObject, weakrefs);

    TF_RET_CHECK(PyType_Ready(type) == 0);
    type_ = reinterpret_cast<PyObject*>(type);
  }
  py::object type = py::reinterpret_borrow<py::object>(type_);
  m.attr("DeviceArray") = type;
  m.attr("PyLocalBuffer") = type;
  m.attr("Buffer") = type;

  // Add methods and properties to the class. We use pybind11 and add methods
  // dynamically mostly because this is easy to write and allows us to use
  // pybind11's casting logic. This is most likely slightly slower than
  // hand-writing bindings, but most of these methods are not performance
  // critical.
  using jax::property;
  using jax::property_readonly;
  type.attr("__array_priority__") =
      property_readonly([](py::object self) -> int { return 100; });
  type.attr("_device") = property(
      [](PyBuffer::object self) -> ClientAndPtr<PjRtDevice> {
        return WrapWithClient(self.buf()->client(),
                              self.buf()->sticky_device());
      },
      [](PyBuffer::object self, PjRtDevice* sticky_device) {
        return self.buf()->set_sticky_device(sticky_device);
      });
  type.attr("aval") = property(
      [](PyBuffer::object self) -> py::object { return self.buf()->GetAval(); },
      [](PyBuffer::object self, py::object aval) {
        return self.buf()->SetAval(std::move(aval));
      });
  type.attr("weak_type") = property(
      [](PyBuffer::object self) -> std::optional<bool> {
        return self.buf()->weak_type();
      },
      [](PyBuffer::object self, std::optional<bool> weak_type) {
        return self.buf()->set_weak_type(weak_type);
      });
  type.attr("device_buffer") =
      property_readonly([](py::object self) { return self; });
  type.attr(
      "shape") = property_readonly([](PyBuffer::object self) -> py::tuple {
    return SpanToTuple(self.buf()->buffer()->on_device_shape().dimensions());
  });
  type.attr("dtype") = property_readonly([](PyBuffer::object self) {
    PrimitiveType primitive =
        self.buf()->buffer()->on_device_shape().element_type();
    return PrimitiveTypeToDtype(primitive).ValueOrDie();
  });
  type.attr("size") =
      property_readonly([](PyBuffer::object self) -> StatusOr<int64_t> {
        return self.buf()->size();
      });
  type.attr("ndim") = property_readonly(
      [](PyBuffer::object self) -> int { return self.buf()->ndim(); });
  type.attr("_value") = property_readonly(
      [](PyBuffer::object self) -> StatusOr<pybind11::object> {
        GlobalPyRefManager()->CollectGarbage();
        return self.buf()->AsNumPyArray(self);
      });
  type.attr("copy_to_device") = py::cpp_function(
      [](PyBuffer::object self, const ClientAndPtr<PjRtDevice>& dst_device) {
        return self.buf()->CopyToDevice(dst_device);
      },
      py::is_method(type));
  type.attr("copy_to_remote_device") = py::cpp_function(
      [](PyBuffer::object self, const py::bytes serialized_descriptor) {
        // TODO(phawkins): remove the std::string cast after C++17 is required.
        // py::bytes has a std::string_view cast, but not an absl::string_view
        // cast.
        return self.buf()->CopyToRemoteDevice(
            static_cast<std::string>(serialized_descriptor));
      },
      py::is_method(type));

  type.attr("on_device_size_in_bytes") = py::cpp_function(
      [](PyBuffer::object self) -> StatusOr<size_t> {
        return self.buf()->OnDeviceSizeInBytes();
      },
      py::is_method(type));
  type.attr("delete") = py::cpp_function(
      [](PyBuffer::object self) { self.buf()->Delete(); }, py::is_method(type));
  type.attr("block_host_until_ready") = py::cpp_function(
      [](PyBuffer::object self) {
        // TODO(phawkins): remove 3 months after the release of jaxlib >= 0.3.2.
        PythonDeprecationWarning(
            "block_host_until_ready() on a JAX array object is deprecated, use "
            "block_until_ready() instead.");
        return self.buf()->BlockHostUntilReady();
      },
      py::is_method(type));
  type.attr("is_ready") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->IsReady(); },
      py::is_method(type));
  type.attr("is_known_ready") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->IsKnownReady(); },
      py::is_method(type));
  type.attr("block_until_ready") = py::cpp_function(
      [](PyBuffer::object self) -> StatusOr<PyBuffer::object> {
        TF_RETURN_IF_ERROR(self.buf()->BlockHostUntilReady());
        return std::move(self);
      },
      py::is_method(type));
  type.attr("copy_to_host_async") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->CopyToHostAsync(); },
      py::is_method(type));
  type.attr("to_py") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->AsNumPyArray(self); },
      py::is_method(type));
  type.attr("xla_shape") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->shape(); },
      py::is_method(type));
  type.attr("xla_dynamic_shape") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->xla_dynamic_shape(); },
      py::is_method(type));
  type.attr("client") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->client(); });
  type.attr("device") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->device(); },
      py::is_method(type));
  type.attr("platform") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->platform_name(); },
      py::is_method(type));
  type.attr("is_deleted") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->is_deleted(); },
      py::is_method(type));
  type.attr("unsafe_buffer_pointer") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->UnsafeBufferPointer(); },
      py::is_method(type));
  type.attr("__cuda_array_interface__") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->CudaArrayInterface(); });
  type.attr("traceback") = property_readonly(
      [](PyBuffer::object self) { return self.buf()->traceback(); });
  type.attr("clone") = py::cpp_function(
      [](PyBuffer::object self) { return self.buf()->Clone(); },
      py::is_method(type));
  type.attr("__module__") = m.attr("__name__");
  return OkStatus();
}

}  // namespace xla
