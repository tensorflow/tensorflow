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

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/device.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/compiler/xla/python/python_utils.h"
#include "tensorflow/compiler/xla/python/transfer_guard_lib.h"
#include "tensorflow/compiler/xla/python/types.h"
#include "tensorflow/compiler/xla/python/util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
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

// Returns if shape has a major-to-minor layout.
bool HasMajorToMinorLayout(const xla::Shape& shape) {
  if (shape.has_layout()) {
    for (int i = 0; i < shape.layout().minor_to_major_size(); ++i) {
      if (shape.layout().minor_to_major(i) !=
          shape.layout().minor_to_major_size() - 1 - i) {
        return false;
      }
    }
  }
  return true;
}

// Returns byte_strides if shape has a non-major-to-minor layout.
std::optional<std::vector<int64_t>> ByteStridesOrDefaultForShapeInt64(
    const Shape& shape) {
  if (!shape.has_layout() || HasMajorToMinorLayout(shape)) {
    return std::nullopt;
  }
  return ByteStridesForShapeInt64(shape);
}

}  // namespace

/*static*/ PyBuffer::object PyBuffer::Make(
    std::shared_ptr<PyClient> client, tsl::RCReference<ifrt::Array> ifrt_array,
    std::shared_ptr<Traceback> traceback) {
  py::object obj = py::reinterpret_steal<py::object>(PyBuffer_tp_new(
      reinterpret_cast<PyTypeObject*>(type_), nullptr, nullptr));
  PyBufferPyObject* buf = reinterpret_cast<PyBufferPyObject*>(obj.ptr());
  new (&buf->buffer)
      PyBuffer(std::move(client), std::move(ifrt_array), std::move(traceback));
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
    return InvalidArgument("Expected a DeviceArray, got object of type %s",
                           py::cast<std::string>(py::str(handle.get_type())));
  }
  return AsPyBufferUnchecked(handle);
}

py::handle PyBuffer::AsHandle() {
  return reinterpret_cast<PyObject*>(reinterpret_cast<char*>(this) -
                                     offsetof(PyBufferPyObject, buffer));
}

PyBuffer::PyBuffer(std::shared_ptr<PyClient> client,
                   tsl::RCReference<ifrt::Array> ifrt_array,
                   std::shared_ptr<Traceback> traceback)
    : client_(std::move(client)),
      ifrt_array_(std::move(ifrt_array)),
      traceback_(std::move(traceback)) {
  CHECK(PyGILState_Check());
  const int device_id = ifrt_array_->sharding().devices().front()->id();
  if (device_id >= client_->buffers_.size()) {
    client_->buffers_.resize(device_id + 1);
  }
  next_ = client_->buffers_[device_id];
  client_->buffers_[device_id] = this;
  prev_ = nullptr;
  if (next_) {
    next_->prev_ = this;
  }
}

PyBuffer::~PyBuffer() {
  CHECK(PyGILState_Check());
  const int device_id = ifrt_array_->sharding().devices().front()->id();
  if (client_->buffers_[device_id] == this) {
    client_->buffers_[device_id] = next_;
  }
  if (prev_) {
    prev_->next_ = next_;
  }
  if (next_) {
    next_->prev_ = prev_;
  }
}

StatusOr<int64_t> PyBuffer::size() {
  if (llvm::isa<ifrt::PjRtCompatibleArray>(ifrt_array_.get())) {
    Shape max_buffer_shape = pjrt_buffer()->on_device_shape();
    if (max_buffer_shape.is_dynamic()) {
      TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
      return ShapeUtil::ElementsIn(*dynamic_shape);
    }
    return ShapeUtil::ElementsIn(max_buffer_shape);
  } else {
    return ifrt_array_->shape().num_elements();
  }
}

/* static */ PjRtBuffer* IfrtHelpers::pjrt_buffer(ifrt::Array* ifrt_array) {
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr == nullptr) {
    throw XlaRuntimeError(
        "This operation is implemented for a PjRt-compatible backend only.");
  }
  return arr->pjrt_buffers().front().get();
}

/* static */ PjRtDevice* IfrtHelpers::pjrt_device(ifrt::Array* ifrt_array) {
  return ifrt_array->sharding().devices().front();
}

/* static */ StatusOr<const Shape*> IfrtHelpers::xla_dynamic_shape(
    ifrt::Array* ifrt_array, std::optional<Shape>& scratch) {
  auto* pjrt_buffer = IfrtHelpers::pjrt_buffer(ifrt_array);

  if (pjrt_buffer->on_device_shape().is_static()) {
    return &pjrt_buffer->on_device_shape();
  }
  // Python buffer protocol references shape data by pointer, therefore we must
  // store a valid copy of the shape.
  if (!scratch) {
    Shape dynamic_shape;
    {
      py::gil_scoped_release gil_release;
      TF_ASSIGN_OR_RETURN(dynamic_shape,
                          pjrt_buffer->logical_on_device_shape());
    }
    scratch = dynamic_shape;
  }
  return &scratch.value();
}

StatusOr<const Shape*> PyBuffer::xla_dynamic_shape() {
  CHECK(PyGILState_Check());
  return IfrtHelpers::xla_dynamic_shape(ifrt_array(), dynamic_shape_);
}

pybind11::tuple IfrtHelpers::python_shape(ifrt::Array* ifrt_array) {
  return SpanToTuple(ifrt_array->shape().dims());
}

pybind11::dtype IfrtHelpers::python_dtype(ifrt::Array* ifrt_array) {
  // TODO(hyeontaek): Support non-XLA types such as xla::ifrt::DType::kString.
  PrimitiveType primitive = ifrt::ToPrimitiveType(ifrt_array->dtype()).value();
  return PrimitiveTypeToDtype(primitive).value();
}

ClientAndPtr<PjRtDevice> PyBuffer::device() const {
  return WrapWithClient(client_, ifrt_array_->sharding().devices().front());
}

PyBuffer::object PyBuffer::Clone() const {
  auto buffer = Make(client_,
                     ifrt_array_
                         ->Reshard(ifrt_array_->shared_ptr_sharding(),
                                   ifrt::ArrayCopySemantics::kReuseInput)
                         .value(),
                     traceback_);
  buffer.buf()->sticky_device_ = sticky_device_;
  buffer.buf()->aval_ = aval_;
  return buffer;
}

/* static */ StatusOr<tsl::RCReference<ifrt::Array>> IfrtHelpers::CopyToDevice(
    ifrt::Array* ifrt_array, PjRtDevice* dst_device) {
  CHECK(dst_device != nullptr);
  auto transfer_guard_formatter = [ifrt_array, dst_device] {
    auto shape = py::cast<std::string>(py::str(python_shape(ifrt_array)));
    auto dtype = py::cast<std::string>(py::str(python_dtype(ifrt_array)));
    return absl::StrCat("shape=", shape, ", dtype=", dtype,
                        ", device=", pjrt_device(ifrt_array)->DebugString(),
                        ", dst_device=", dst_device->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToDevice(transfer_guard_formatter));

  GlobalPyRefManager()->CollectGarbage();
  py::gil_scoped_release gil_release;
  return ifrt_array->Reshard(ifrt::SingleDeviceSharding::Create(dst_device),
                             ifrt::ArrayCopySemantics::kReuseInput);
}

StatusOr<py::object> PyBuffer::CopyToDevice(
    const ClientAndPtr<PjRtDevice>& dst_device) const {
  TF_ASSIGN_OR_RETURN(
      tsl::RCReference<ifrt::Array> out,
      IfrtHelpers::CopyToDevice(ifrt_array(), dst_device.get()));
  auto traceback = Traceback::Get();
  return Make(dst_device.client, std::move(out), std::move(traceback));
}

std::pair<Status, bool> PyBuffer::CopyToRemoteDevice(
    absl::string_view serialized_descriptor) const {
  absl::Mutex mu;
  bool done = false;
  Status status;
  bool sends_were_enqueued;
  pjrt_buffer()->CopyToRemoteDevice(
      PjRtFuture<StatusOr<std::string>>(std::string(serialized_descriptor)),
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
  return AwaitBuffersReady(ifrt_array_.get());
}

/* static */ StatusOr<pybind11::object> PyHostValue::AsNumPyArray(
    std::shared_ptr<PyHostValue>& host_value,
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array,
    pybind11::handle this_obj) {
  if (ifrt_array->IsDeleted()) {
    return InvalidArgument("DeviceArray has been deleted.");
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    TF_RET_CHECK(pjrt_buffer->on_device_shape().IsArray());
    // On CPU, we can return the value in a zero-copy way.
    if (pjrt_buffer->IsOnCpu()) {
      TF_ASSIGN_OR_RETURN(
          const auto* shape,
          IfrtHelpers::xla_dynamic_shape(ifrt_array, dynamic_shape_holder));
      TF_ASSIGN_OR_RETURN(py::dtype dtype,
                          PrimitiveTypeToDtype(shape->element_type()));
      // Objects that must be kept alive while the array is alive.
      struct Hold {
        tsl::RCReference<ifrt::Array> buffer;
        std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold;
      };
      auto hold = std::make_unique<Hold>();
      TF_ASSIGN_OR_RETURN(hold->external_reference_hold,
                          pjrt_buffer->AcquireExternalReference());
      hold->buffer = tsl::FormRef(ifrt_array);
      void* data =
          hold->external_reference_hold->OpaqueDeviceMemoryDataPointer();
      py::capsule hold_capsule(hold.release(),
                               [](void* h) { delete static_cast<Hold*>(h); });
      py::array array(dtype, shape->dimensions(), ByteStridesForShape(*shape),
                      data, hold_capsule);
      array.attr("flags").attr("writeable") = Py_False;
      {
        py::gil_scoped_release gil;
        TF_RETURN_IF_ERROR(ifrt_array->GetReadyFuture().Await());
      }
      return array;
    }
  }

  TF_RETURN_IF_ERROR(
      CopyToHostAsync(host_value, dynamic_shape_holder, ifrt_array));
  if (!host_value->ready.HasBeenNotified()) {
    py::gil_scoped_release gil;
    host_value->ready.WaitForNotification();
  }
  TF_RETURN_IF_ERROR(host_value->status);
  TF_ASSIGN_OR_RETURN(py::object array, LiteralToPython(host_value->value));
  array.attr("flags").attr("writeable") = Py_False;
  return array;
}

/* static */ Status PyHostValue::CopyToHostAsync(
    std::shared_ptr<PyHostValue>& host_value,
    std::optional<Shape>& dynamic_shape_holder, ifrt::Array* ifrt_array) {
  if (host_value) {
    return OkStatus();
  }
  auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array);
  if (arr != nullptr) {
    auto* pjrt_buffer = arr->pjrt_buffers().front().get();
    if (pjrt_buffer->IsOnCpu()) {
      return OkStatus();
    }
  }
  auto transfer_guard_formatter = [ifrt_array] {
    auto shape =
        py::cast<std::string>(py::str(IfrtHelpers::python_shape(ifrt_array)));
    auto dtype =
        py::cast<std::string>(py::str(IfrtHelpers::python_dtype(ifrt_array)));
    return absl::StrCat("shape=", shape, ", dtype=", dtype, ", device=",
                        IfrtHelpers::pjrt_device(ifrt_array)->DebugString());
  };
  TF_RETURN_IF_ERROR(
      jax::ApplyTransferGuardToDeviceToHost(transfer_guard_formatter));

  auto host_value_copy = std::make_shared<PyHostValue>();
  host_value = host_value_copy;
  // TODO(b/182461453): This is a blocking call. If we further implemented
  // populating dynamic shape metadata while fetching the literal, we wouldn't
  // need this static approach.
  const xla::Shape* dynamic_shape;
  std::optional<xla::Shape> shape_holder;
  if (llvm::isa<ifrt::PjRtCompatibleArray>(ifrt_array)) {
    TF_ASSIGN_OR_RETURN(dynamic_shape, IfrtHelpers::xla_dynamic_shape(
                                           ifrt_array, dynamic_shape_holder));
  } else {
    // Skip querying the dynamic shape for a non-PjRt Array.
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType type,
                        ifrt::ToPrimitiveType(ifrt_array->dtype()));
    shape_holder = ShapeUtil::MakeShapeWithDescendingLayout(
        type, ifrt_array->shape().dims());
    dynamic_shape = &*shape_holder;
  }

  py::gil_scoped_release gil;
  xla::Shape host_shape = ShapeUtil::DeviceShapeToHostShape(*dynamic_shape);
  // TODO(hyeontaek): Several PjRt runtimes assume that the host buffer uses
  // the same transposition as the device buffer. This is different from
  // PjRtBuffer::ToLiteral()'s semantics that the runtime respects the layout
  // of the host buffer literal. On the other hand, the runtime often knows
  // better about an efficient layout for the host buffer. It will be useful
  // to revisit the semantics of PjRtBuffer::ToLiteral() to see if it is
  // desirable for the runtime to choose the layout.
  host_value_copy->value = std::make_shared<Literal>(host_shape);
  ifrt::Future<Status> copy_future = ifrt_array->CopyToHostBuffer(
      host_value_copy->value->untyped_data(),
      ByteStridesOrDefaultForShapeInt64(host_shape),
      ifrt::ArrayCopySemantics::kReuseInput);
  copy_future.OnReady([host_value{std::move(host_value_copy)}](Status status) {
    host_value->status = std::move(status);
    host_value->ready.Notify();
  });
  return OkStatus();
}

Status PyBuffer::CopyToHostAsync() {
  return PyHostValue::CopyToHostAsync(host_value_, dynamic_shape_,
                                      ifrt_array_.get());
}

StatusOr<pybind11::object> PyBuffer::AsNumPyArray(py::handle this_obj) {
  return PyHostValue::AsNumPyArray(host_value_, dynamic_shape_,
                                   ifrt_array_.get(), this_obj);
}

StatusOr<std::uintptr_t> PyBuffer::UnsafeBufferPointer() const {
  return client_->pjrt_client()->UnsafeBufferPointer(pjrt_buffer());
}

StatusOr<py::dict> PyBuffer::CudaArrayInterface() {
  // TODO(zhangqiaorjc): Differentiate between NVidia and other GPUs.
  if (pjrt_buffer()->client()->platform_id() != GpuId()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for NVidia GPU buffers.");
  }
  if (!pjrt_buffer()->on_device_shape().IsArray()) {
    return InvalidArgument(
        "__cuda_array_interface__ is only defined for array buffers.");
  }
  if (pjrt_buffer()->on_device_shape().element_type() == BF16) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for bfloat16 buffers.");
  }
  if (pjrt_buffer()->on_device_shape().element_type() == F8E4M3FN) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for F8E4M3FN buffers.");
  }
  if (pjrt_buffer()->on_device_shape().element_type() == F8E5M2) {
    return InvalidArgument(
        "__cuda_array_interface__ is not supported for F8E5M2 buffers.");
  }
  TF_RET_CHECK(LayoutUtil::IsMonotonicWithDim0Major(
      pjrt_buffer()->on_device_shape().layout()));

  py::dict result;
  TF_ASSIGN_OR_RETURN(const auto* dynamic_shape, xla_dynamic_shape());
  result["shape"] = SpanToTuple(dynamic_shape->dimensions());
  TF_ASSIGN_OR_RETURN(py::str typestr,
                      TypeDescriptorForPrimitiveType(
                          pjrt_buffer()->on_device_shape().element_type()));
  result["typestr"] = std::move(typestr);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtBuffer::ExternalReference> external_reference_hold,
      pjrt_buffer()->AcquireExternalReference());
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
    PjRtBuffer* buffer_ptr;
    try {
      buffer_ptr = py_buffer->pjrt_buffer();
    } catch (const XlaRuntimeError& e) {
      return InvalidArgument("%s", e.what());
    }

    PjRtBuffer& buffer = *buffer_ptr;
    if (!buffer.IsOnCpu()) {
      return InvalidArgument(
          "Python buffer protocol is only defined for CPU buffers.");
    }

    TF_ASSIGN_OR_RETURN(const auto* shape, py_buffer->xla_dynamic_shape());
    // Py_buffer objects are POD C structures, so we don't need to hold the GIL.
    // Additionally we call BlockHostUntilReady() below, which may block.
    py::gil_scoped_release gil_release;

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
    if (buffer.on_device_shape().element_type() == F8E4M3FN &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "F8E4M3FN buffer format not supported by Python buffer protocol.");
    }
    if (buffer.on_device_shape().element_type() == F8E5M2 &&
        ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)) {
      return InvalidArgument(
          "F8E5M2 buffer format not supported by Python buffer protocol.");
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
  type.attr("__array__") = py::cpp_function(
      [](PyBuffer::object self, py::object dtype, py::object context) {
        py::object array = ValueOrThrow(self.buf()->AsNumPyArray(self));
        if (!dtype.is_none()) {
          return array.attr("astype")(dtype);
        }
        return array;
      },
      py::is_method(type), py::arg("dtype") = py::none(),
      py::arg("context") = py::none());
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
  type.attr("shape") =
      property_readonly([](PyBuffer::object self) -> py::tuple {
        return SpanToTuple(self.buf()->ifrt_array()->shape().dims());
      });
  type.attr("dtype") =
      property_readonly([](PyBuffer::object self) -> StatusOr<py::dtype> {
        TF_ASSIGN_OR_RETURN(
            auto primitive_type,
            ifrt::ToPrimitiveType(self.buf()->ifrt_array()->dtype()));
        return PrimitiveTypeToDtype(primitive_type);
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

StatusOr<ifrt::DType> ToIfRtDType(py::dtype dtype) {
  TF_ASSIGN_OR_RETURN(auto primitive_type, DtypeToPrimitiveType(dtype));
  return ifrt::ToDType(primitive_type);
}

StatusOr<py::dtype> ToPybind11DType(ifrt::DType dtype) {
  TF_ASSIGN_OR_RETURN(auto primitive_type, ifrt::ToPrimitiveType(dtype));
  return PrimitiveTypeToDtype(primitive_type);
}

}  // namespace xla
