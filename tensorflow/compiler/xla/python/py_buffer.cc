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

#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

namespace py = pybind11;

PyBuffer::PyBuffer(std::shared_ptr<PyClient> client,
                   std::unique_ptr<PjRtBuffer> buffer,
                   std::shared_ptr<Traceback> traceback)
    : client_(std::move(client)),
      buffer_(std::move(buffer)),
      traceback_(std::move(traceback)) {
  next_ = client_->buffers_;
  client_->buffers_ = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyBuffer::~PyBuffer() {
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

ClientAndPtr<Device> PyBuffer::device() const {
  return WrapWithClient(client_, buffer_->device());
}

StatusOr<std::unique_ptr<PyBuffer>> PyBuffer::CopyToDevice(
    const ClientAndPtr<Device>& dst_device) const {
  CHECK(dst_device.get() != nullptr);
  GlobalPyRefManager()->CollectGarbage();
  auto traceback = Traceback::Get();
  py::gil_scoped_release gil_release;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtBuffer> out,
                      buffer_->CopyToDevice(dst_device.get()));
  return std::make_unique<PyBuffer>(dst_device.client, std::move(out),
                                    std::move(traceback));
}

Status PyBuffer::BlockHostUntilReady() {
  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return buffer_->BlockHostUntilReady();
}

StatusOr<std::uintptr_t> PyBuffer::UnsafeBufferPointer() const {
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, buffer_->AsShapedBuffer());
  if (shaped_buffer.on_device_shape().IsTuple()) {
    return Unimplemented(
        "unsafe_buffer_pointer is not implemented for tuple "
        "buffers.");
  }
  return absl::bit_cast<std::uintptr_t>(shaped_buffer.root_buffer().opaque());
}

StatusOr<py::dict> PyBuffer::CudaArrayInterface() const {
  if (buffer_->device()->local_device_state()->executor()->platform_kind() !=
      se::PlatformKind::kCuda) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!buffer_->on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (buffer_->on_host_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  TF_RET_CHECK(
      LayoutUtil::IsMonotonicWithDim0Major(buffer_->on_host_shape().layout()));
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, buffer_->AsShapedBuffer());

  py::dict result;
  result["shape"] = IntSpanToTuple(shaped_buffer.on_host_shape().dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          shaped_buffer.on_host_shape().element_type()));
  result["typestr"] = std::move(typestr);
  py::tuple data(2);
  data[0] = py::int_(
      absl::bit_cast<std::uintptr_t>(shaped_buffer.root_buffer().opaque()));
  data[1] = py::bool_(true);  // read-only
  result["data"] = std::move(data);
  result["version"] = py::int_(2);
  return result;
}

// PEP 3118 buffer protocol implementation.

namespace {

// Extra data to be kept alive by the consumer of the buffer protocol.
struct ExtraBufferInfo {
  explicit ExtraBufferInfo(PjRtBuffer::ScopedHold device_buffer)
      : device_buffer(std::move(device_buffer)) {}

  std::string format;
  std::vector<Py_ssize_t> strides;
  // We keep a reference to the TrackedDeviceBuffer that backs the
  // PjRtBuffer. This prevents a use-after-free in the event that Delete() is
  // called on a buffer with an live buffer protocol view. It does however mean
  // that Delete() sometimes won't actually delete immediately.
  PjRtBuffer::ScopedHold device_buffer;
};

int PjRtBufferGetBuffer(PyObject* exporter, Py_buffer* view, int flags) {
  auto& buffer =
      *py::reinterpret_borrow<py::object>(exporter).cast<PyBuffer&>().buffer();
  Status status = [&]() {
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

    if (buffer.device()->platform_name() != "cpu") {
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
    if (buffer.on_host_shape().element_type() == BF16 &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "bfloat16 buffer format not supported by Python buffer protocol.");
    }
    if ((flags & PyBUF_WRITEABLE) == PyBUF_WRITEABLE) {
      return InvalidArgument("XLA buffers are read-only.");
    }
    PjRtBuffer::ScopedHold device_buffer(
        buffer.GetBufferWithExternalReference());
    if (!device_buffer.status().ok()) {
      return InvalidArgument("Deleted buffer used in buffer protocol.");
    }
    const Shape& shape = buffer.on_host_shape();
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
    CHECK_EQ(device_buffer->device_memory().size(), 1);
    view->buf =
        const_cast<void*>(device_buffer->device_memory().front().opaque());
    auto extra = absl::make_unique<ExtraBufferInfo>(std::move(device_buffer));
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

}  // namespace xla
