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

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace py = pybind11;

PyBuffer::PyBuffer(std::shared_ptr<PyClient> client,
                   std::shared_ptr<PjRtBuffer> buffer,
                   std::shared_ptr<Traceback> traceback)
    : client_(std::move(client)),
      buffer_(std::move(buffer)),
      traceback_(std::move(traceback)) {
  CHECK(PyGILState_Check());
  next_ = client_->buffers_;
  client_->buffers_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyBuffer::~PyBuffer() {
  CHECK(PyGILState_Check());
  if (client_->buffers_ == this) {
    client_->buffers_ = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

StatusOr<int64> PyBuffer::size() {
  Shape max_buffer_shape = buffer()->on_device_shape();
  if (max_buffer_shape.is_dynamic()) {
    TF_ASSIGN_OR_RETURN(auto dynamic_shape, xla_dynamic_shape());
    return ShapeUtil::ElementsIn(dynamic_shape);
  }
  return ShapeUtil::ElementsIn(max_buffer_shape);
}

StatusOr<Shape> PyBuffer::xla_dynamic_shape() {
  DCHECK(PyGILState_Check());
  if (buffer_->on_device_shape().is_static()) {
    return buffer_->on_device_shape();
  }
  if (!dynamic_shape_) {
    Shape dynamic_shape;
    {
      py::gil_scoped_release gil_release;
      TF_ASSIGN_OR_RETURN(dynamic_shape, buffer_->logical_on_device_shape());
    }
    dynamic_shape_ = dynamic_shape;
  }
  return dynamic_shape_.value();
}

pybind11::tuple PyBuffer::python_shape() const {
  return IntSpanToTuple(buffer()->on_device_shape().dimensions());
}

pybind11::dtype PyBuffer::python_dtype() const {
  PrimitiveType primitive = buffer()->on_device_shape().element_type();
  return PrimitiveTypeToDtype(primitive).ValueOrDie();
}

ClientAndPtr<PjRtDevice> PyBuffer::device() const {
  return WrapWithClient(client_, buffer_->device());
}

std::unique_ptr<PyBuffer> PyBuffer::Clone() const {
  auto buffer = std::make_unique<PyBuffer>(client_, buffer_, traceback_);
  buffer->sticky_device_ = sticky_device_;
  buffer->aval_ = aval_;
  return buffer;
}

StatusOr<std::unique_ptr<PyBuffer>> PyBuffer::CopyToDevice(
    const ClientAndPtr<PjRtDevice>& dst_device) const {
  CHECK(dst_device.get() != nullptr);
  GlobalPyRefManager()->CollectGarbage();
  std::unique_ptr<PjRtBuffer> out;
  {
    py::gil_scoped_release gil_release;
    TF_ASSIGN_OR_RETURN(out, buffer_->CopyToDevice(dst_device.get()));
  }
  auto traceback = Traceback::Get();
  return std::make_unique<PyBuffer>(dst_device.client, std::move(out),
                                    std::move(traceback));
}

Status PyBuffer::BlockHostUntilReady() {
  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return buffer_->BlockHostUntilReady();
}

Status PyBuffer::CopyToHostAsync() {
  if (!buffer_->IsOnCpu() && !host_value_) {
    std::shared_ptr<HostValue> host_value = std::make_shared<HostValue>();
    host_value_ = host_value;
    // TODO(b/182461453): This is a blocking call. If we further implemented
    // populating dynamic shape metadata while fetching the literal, we wouldn't
    // need this static approach.
    TF_ASSIGN_OR_RETURN(Shape dynamic_shape, xla_dynamic_shape());

    py::gil_scoped_release gil;
    host_value->value = std::make_shared<Literal>(
        ShapeUtil::DeviceShapeToHostShape(dynamic_shape));
    Literal* literal = host_value->value.get();
    buffer_->ToLiteral(literal,
                       [host_value{std::move(host_value)}](Status status) {
                         host_value->status = std::move(status);
                         host_value->ready.Notify();
                       });
  }
  return Status::OK();
}

StatusOr<pybind11::object> PyBuffer::AsNumPyArray(py::handle this_obj) {
  if (buffer_->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  TF_RET_CHECK(buffer_->on_device_shape().IsArray());
  // On CPU, we can return the value in a zero-copy way.
  if (buffer_->IsOnCpu()) {
    Shape shape = buffer_->on_device_shape();
    if (shape.is_dynamic()) {
      return Unimplemented("DynamicShape not implemented for CPU device.");
    }
    TF_ASSIGN_OR_RETURN(py::dtype dtype,
                        PrimitiveTypeToDtype(shape.element_type()));
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
    py::array array(dtype, shape.dimensions(), ByteStridesForShape(shape), data,
                    hold_capsule);
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
  if (buffer_->client()->platform_id() != kGpuId) {
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
  TF_ASSIGN_OR_RETURN(Shape dynamic_shape, xla_dynamic_shape());
  result["shape"] = IntSpanToTuple(dynamic_shape.dimensions());
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

int PjRtBufferGetBuffer(PyObject* exporter, Py_buffer* view, int flags) {
  auto& buffer =
      *py::reinterpret_borrow<py::object>(exporter).cast<PyBuffer&>().buffer();
  Status status = [&]() {
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (!buffer.IsOnCpu()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }
    if (buffer.on_device_shape().is_dynamic()) {
      return InvalidArgument(
          "Dynamic shape is not supported for Python buffer protocol.");
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
    const Shape& shape = buffer.on_device_shape();
    if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS ||
         (flags & PyBUF_STRIDES) == PyBUF_ND) &&
        !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
      return InvalidArgument("Buffer is not in C-contiguous layout.");
    } else if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape.layout())) {
      return InvalidArgument("Buffer is not in F-contiguous layout.");
    } else if ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS &&
               !LayoutUtil::IsMonotonicWithDim0Major(shape.layout()) &&
               !LayoutUtil::IsMonotonicWithDim0Minor(shape.layout())) {
      return InvalidArgument("Buffer is not in contiguous layout.");
    }
    std::memset(view, 0, sizeof(Py_buffer));
    const void* root_ptr =
        external_reference_hold->OpaqueDeviceMemoryDataPointer();
    view->buf = const_cast<void*>(root_ptr);
    auto extra =
        absl::make_unique<ExtraBufferInfo>(std::move(external_reference_hold));
    view->itemsize = ShapeUtil::ByteSizeOfPrimitiveType(shape.element_type());
    view->len = ShapeUtil::ByteSizeOf(shape);
    view->readonly = 1;
    if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT) {
      TF_ASSIGN_OR_RETURN(extra->format, FormatDescriptorForPrimitiveType(
                                             shape.element_type()));
      view->format = const_cast<char*>(extra->format.c_str());
    }
    if ((flags & PyBUF_ND) == PyBUF_ND) {
      view->ndim = shape.dimensions_size();
      static_assert(sizeof(int64) == sizeof(Py_ssize_t),
                    "Py_ssize_t must be 64 bits");
      if (view->ndim != 0) {
        view->shape = reinterpret_cast<Py_ssize_t*>(
            const_cast<int64*>(shape.dimensions().data()));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          extra->strides = ByteStridesForShape(shape);
          view->strides = extra->strides.data();
        }
      }
    }
    TF_RETURN_IF_ERROR(buffer.BlockHostUntilReady());
    view->internal = extra.release();
    return Status::OK();
  }();
  if (!status.ok()) {
    PyErr_SetString(PyExc_BufferError, status.ToString().c_str());
    return -1;
  }
  view->obj = exporter;
  Py_INCREF(view->obj);
  return 0;
}

void PjRtBufferReleaseBuffer(PyObject*, Py_buffer* buffer) {
  auto extra = static_cast<ExtraBufferInfo*>(buffer->internal);
  delete extra;
}

PyBufferProcs PjRtBufferProcs = []() {
  PyBufferProcs procs;
  procs.bf_getbuffer = &PjRtBufferGetBuffer;
  procs.bf_releasebuffer = &PjRtBufferReleaseBuffer;
  return procs;
}();

}  // namespace

/*static*/ PyBufferProcs* PyBuffer::BufferProtocol() {
  return &PjRtBufferProcs;
}

void PyBuffer::SetStickyDevice(pybind11::object sticky_device) {
  if (sticky_device_ && !sticky_device_->equal(sticky_device)) {
    throw std::invalid_argument(
        "One cannot set again the stickyness of a buffer and needs to create "
        "a new one or a `_DeviceArray`");
  }
  sticky_device_ = sticky_device;
}

void PyBuffer::SetAval(pybind11::object aval) {
  if (aval_ && !aval_->equal(aval)) {
    throw std::invalid_argument(
        "One cannot set again the aval_ of a buffer and needs to create a "
        "new one or a `_DeviceArray`");
  }
  aval_ = aval;
}

}  // namespace xla
