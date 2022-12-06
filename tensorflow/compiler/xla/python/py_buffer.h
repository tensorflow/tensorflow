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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#ifdef JAX_ENABLE_IFRT
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#endif
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Python wrapper around PjRtBuffer. We use a wrapper class:
// a) to keep the PjRtClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
//
// A `PyBuffer` can be used from Python without being wrapped in a Python
// `DeviceArray` object.
class PyBuffer {
 public:
  // pybind11::object typed subclass for PyBuffer objects.
  class pyobject : public pybind11::object {
   public:
    PYBIND11_OBJECT(pyobject,  // NOLINT
                    pybind11::object, PyBuffer::IsPyBuffer);
    pyobject() = default;
    PyBuffer* buf() const { return PyBuffer::AsPyBufferUnchecked(*this); }
  };
  using object = pyobject;

#ifdef JAX_ENABLE_IFRT
  static object Make(std::shared_ptr<PyClient> client,
                     std::unique_ptr<ifrt::Array> ifrt_array,
                     std::shared_ptr<Traceback> traceback);
#else
  static object Make(std::shared_ptr<PyClient> client,
                     std::shared_ptr<PjRtBuffer> buffer,
                     std::shared_ptr<Traceback> traceback);
#endif

  // Returns true if `h` is a PyBuffer.
  static bool IsPyBuffer(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Does not do any checking.
  static PyBuffer* AsPyBufferUnchecked(pybind11::handle handle);
  // Converts `handle` to a PyBuffer*. Returns an error status if
  // !IsPyBuffer(handle)
  static StatusOr<PyBuffer*> AsPyBuffer(pybind11::handle handle);

  // Gets a Python handle to an existing PyBuffer. Assumes the PyObject was
  // allocated on the Python heap, which is the case if Make() was used.
  pybind11::handle AsHandle();

  ~PyBuffer();

  std::shared_ptr<PyClient> client() const { return client_; }

#ifdef JAX_ENABLE_IFRT
  ifrt::Array* ifrt_array() const { return ifrt_array_.get(); }

  // Short-term escape hatch to get PjRtBuffer from PyBuffer.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  PjRtBuffer* pjrt_buffer() const {
    auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtArray>(ifrt_array_.get());
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return arr->pjrt_buffers().front().get();
  }

  // Short-term escape hatch to get PjRtBuffer from PyBuffer.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  std::shared_ptr<PjRtBuffer> shared_ptr_pjrt_buffer() const {
    auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtArray>(ifrt_array_.get());
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return arr->pjrt_buffers().front();
  }

  void SetPjRtBuffer(std::shared_ptr<PjRtBuffer> buffer) {
    auto ifrt_array =
        xla::ifrt::PjRtArray::Create(client_->ifrt_client(), std::move(buffer));
    TF_CHECK_OK(ifrt_array.status());
    ifrt_array_ = *std::move(ifrt_array);
  }
  void SetPjRtBuffer(std::unique_ptr<PjRtBuffer> buffer) {
    auto ifrt_array =
        xla::ifrt::PjRtArray::Create(client_->ifrt_client(), std::move(buffer));
    TF_CHECK_OK(ifrt_array.status());
    ifrt_array_ = *std::move(ifrt_array);
  }
#else
  PjRtBuffer* pjrt_buffer() const { return buffer_.get(); }

  std::shared_ptr<PjRtBuffer> shared_ptr_pjrt_buffer() const { return buffer_; }

  void SetPjRtBuffer(std::unique_ptr<PjRtBuffer> buffer) {
    buffer_ = std::move(buffer);
  }
#endif

  // Legacy alises.
  PjRtBuffer* buffer() const { return pjrt_buffer(); }
  std::shared_ptr<PjRtBuffer> shared_ptr_buffer() const {
    return shared_ptr_pjrt_buffer();
  }

  ClientAndPtr<PjRtDevice> device() const;
  absl::string_view platform_name() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_array_->client()->platform_name();
#else
    return buffer_->client()->platform_name();
#endif
  }
  bool is_deleted() const {
#ifdef JAX_ENABLE_IFRT
    return ifrt_array_->IsDeleted();
#else
    return buffer_->IsDeleted();
#endif
  }

  StatusOr<pybind11::object> CopyToDevice(
      const ClientAndPtr<PjRtDevice>& dst_device) const;
  std::pair<Status, bool> CopyToRemoteDevice(
      absl::string_view serialized_descriptor) const;

  StatusOr<size_t> OnDeviceSizeInBytes() {
    return pjrt_buffer()->GetOnDeviceSizeInBytes();
  }

  void Delete() {
#ifdef JAX_ENABLE_IFRT
    // TODO(hyeontaek): Return Status.
    TF_CHECK_OK(ifrt_array_->Delete().Await());
#else
    buffer_->Delete();
#endif
    host_value_ = nullptr;
  }

  // Makes a copy of this PyBuffer object that shares the underlying PjRtBuffer.
  // This is useful because we may wish to change JAX metadata (e.g., the sticky
  // device) without copying the buffer.
  object Clone() const;

  // Returns xla::InvalidArgument if the buffer has been deleted.
  // See `PjRtFuture` for the semantics of `IsReady` and `IsKnownReady`.
  StatusOr<bool> IsReady() {
#ifdef JAX_ENABLE_IFRT
    if (ifrt_array_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return ifrt_array_->GetReadyFuture().IsReady();
#else
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsReady();
#endif
  }
  StatusOr<bool> IsKnownReady() {
#ifdef JAX_ENABLE_IFRT
    if (ifrt_array_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return ifrt_array_->GetReadyFuture().IsKnownReady();
#else
    if (buffer_->IsDeleted()) {
      return InvalidArgument("DeviceArray has been deleted.");
    }
    return buffer_->GetReadyFuture().IsKnownReady();
#endif
  }

  // Returns xla::InvalidArgument if the buffer has been deleted.
  Status BlockHostUntilReady();
  Status CopyToHostAsync();

  const Shape& shape() { return pjrt_buffer()->on_device_shape(); }

  StatusOr<std::uintptr_t> UnsafeBufferPointer() const;

  // Implementation of the CUDA array interface for sharing GPU buffers with
  // other Python libraries.
  StatusOr<pybind11::dict> CudaArrayInterface();

  const std::shared_ptr<Traceback>& traceback() const { return traceback_; }

  // Returns the size (i.e. number of elements) of the (host) numpy array.
  StatusOr<int64_t> size();

  // Returns the number of dimensions of the (host) numpy array.
  int ndim() const {
    return pjrt_buffer()->on_device_shape().dimensions_size();
  }

  pybind11::tuple python_shape() const;
  pybind11::dtype python_dtype() const;

  // Representing the logical view of the underlying dynamic shapes.
  StatusOr<const Shape*> xla_dynamic_shape();

  Status set_sticky_device(PjRtDevice* sticky_device) {
#ifdef JAX_ENABLE_IFRT
    TF_RET_CHECK(sticky_device == nullptr ||
                 sticky_device == ifrt_array_->sharding().devices().front());
#else
    TF_RET_CHECK(sticky_device == nullptr ||
                 sticky_device == buffer_->device());
#endif
    sticky_device_ = sticky_device;
    return OkStatus();
  }
  PjRtDevice* sticky_device() const { return sticky_device_; }

  void set_weak_type(std::optional<bool> weak_type) { weak_type_ = weak_type; }
  std::optional<bool> weak_type() const { return weak_type_; }

  StatusOr<pybind11::object> AsNumPyArray(pybind11::handle this_obj);

  void SetAval(pybind11::object aval) { aval_ = aval; }
  pybind11::object GetAval() const { return aval_; }

  static Status RegisterTypes(pybind11::module& m);
  static PyObject* base_type() { return base_type_; }
  static PyObject* type() { return type_; }

 private:
  // PyBuffer objects must not be allocated directly since they must always live
  // on the Python heap. Use Make() instead.
#ifdef JAX_ENABLE_IFRT
  PyBuffer(std::shared_ptr<PyClient> client, std::unique_ptr<ifrt::Array> array,
           std::shared_ptr<Traceback> traceback);
#else
  PyBuffer(std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
           std::shared_ptr<Traceback> traceback);
#endif

  static PyObject* base_type_;
  static PyObject* type_;

  friend class PyClient;

  struct HostValue {
    absl::Notification ready;
    Status status;
    std::shared_ptr<xla::Literal> value;
  };
  std::shared_ptr<PyClient> client_;
#ifdef JAX_ENABLE_IFRT
  std::unique_ptr<ifrt::Array> ifrt_array_;
#else
  std::shared_ptr<PjRtBuffer> buffer_;
#endif
  std::shared_ptr<Traceback> traceback_;
  std::shared_ptr<HostValue> host_value_;  // Protected by the GIL.

  // JAX uses this field to record whether a buffer is committed to a particular
  // device by the user (https://github.com/google/jax/pull/1916).
  PjRtDevice* sticky_device_ = nullptr;

  // TODO(phawkins): consider not keeping an explicit aval on C++ buffer
  // objects.
  pybind11::object aval_ = pybind11::none();

  // An optional weak type. If absent, the JAX jit code computes the weak_type
  // from the aval_.weak_type attribute. This is a backwards compatibility
  // measure for older Python code that does not set weak_type explicitly.
  // TODO(phawkins): drop support for older jax Python versions and make
  // weak_type mandatory.
  std::optional<bool> weak_type_ = std::nullopt;

  std::optional<Shape> dynamic_shape_ = std::nullopt;
  // Doubly-linked list of all PyBuffers known to the client. Protected by the
  // GIL. Since multiple PyBuffers may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyBuffer* next_;
  PyBuffer* prev_;
};

// A batched version of python wrapper around a list of PjRtBuffers.
class PyShardedBuffer {
 public:
  static PyShardedBuffer CreateFromPyBuffers(
      absl::Span<const PyBuffer::object> py_buffers);

#ifdef JAX_ENABLE_IFRT
  PyShardedBuffer(std::shared_ptr<PyClient> client,
                  std::unique_ptr<ifrt::Array> ifrt_array,
                  std::shared_ptr<Traceback> traceback, bool sticky = false)
      : client_(std::move(client)),
        ifrt_array_(std::move(ifrt_array)),
        traceback_(std::move(traceback)),
        sticky_(sticky) {
    Link();
  }
#else
  PyShardedBuffer(std::shared_ptr<PyClient> client,
                  std::vector<std::shared_ptr<PjRtBuffer> > buffers,
                  std::shared_ptr<Traceback> traceback, bool sticky = false)
      : client_(std::move(client)),
        buffers_(std::move(buffers)),
        traceback_(std::move(traceback)),
        sticky_(sticky) {
    Link();
  }
#endif

  PyShardedBuffer(const PyShardedBuffer&) = delete;
  PyShardedBuffer& operator=(const PyShardedBuffer&) = delete;

  PyShardedBuffer(PyShardedBuffer&& other) {
    other.Unlink();
    client_ = std::move(other.client_);
#ifdef JAX_ENABLE_IFRT
    ifrt_array_ = std::move(other.ifrt_array_);
#else
    buffers_ = std::move(other.buffers_);
#endif
    traceback_ = std::move(other.traceback_);
    sticky_ = other.sticky_;
    Link();
  }

  PyShardedBuffer& operator=(PyShardedBuffer&& other) {
    Unlink();
    other.Unlink();
    client_ = std::move(other.client_);
#ifdef JAX_ENABLE_IFRT
    ifrt_array_ = std::move(other.ifrt_array_);
#else
    buffers_ = std::move(other.buffers_);
#endif
    traceback_ = std::move(other.traceback_);
    sticky_ = other.sticky_;
    Link();
    return *this;
  }

  ~PyShardedBuffer() { Unlink(); }

#ifdef JAX_ENABLE_IFRT
  std::vector<PyBuffer::object> GetPyBuffers() const {
    std::vector<PyBuffer::object> results;
    results.reserve(ifrt_array_->sharding().devices().size());
    auto ifrt_arrays = ifrt_array_->DisassembleIntoSingleDeviceArrays(
        ifrt::ArrayCopySemantics::kReuseInput);
    TF_CHECK_OK(ifrt_arrays.status());
    for (auto& ifrt_array : *ifrt_arrays) {
      auto* device = ifrt_array->sharding().devices().front();
      auto py_buffer =
          PyBuffer::Make(client_, std::move(ifrt_array), traceback_);
      if (sticky_) {
        TF_CHECK_OK(py_buffer.buf()->set_sticky_device(device));
      }
      results.push_back(std::move(py_buffer));
    }
    return results;
  }
#else
  std::vector<PyBuffer::object> GetPyBuffers() const {
    std::vector<PyBuffer::object> results;
    results.reserve(buffers_.size());
    for (const auto& pjrt_buffer : buffers_) {
      auto py_buffer = PyBuffer::Make(client_, pjrt_buffer, traceback_);
      if (sticky_) {
        TF_CHECK_OK(py_buffer.buf()->set_sticky_device(pjrt_buffer->device()));
      }
      results.push_back(std::move(py_buffer));
    }
    return results;
  }
#endif

#ifdef JAX_ENABLE_IFRT
  PyBuffer::object GetPyBuffer(int device_id) const {
    // TODO(hyeontaek): Remove this method. This method will not scale well.
    auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtArray>(ifrt_array_.get());
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    auto& pjrt_buffer = arr->pjrt_buffers().at(device_id);
    auto py_buffer = PyBuffer::Make(
        client_,
        ifrt::PjRtArray::Create(client_->ifrt_client(), pjrt_buffer).value(),
        traceback_);
    if (sticky_) {
      TF_CHECK_OK(py_buffer.buf()->set_sticky_device(pjrt_buffer->device()));
    }
    return py_buffer;
  }
#else
  PyBuffer::object GetPyBuffer(int device_id) const {
    const auto& pjrt_buffer = buffers_.at(device_id);
    auto py_buffer = PyBuffer::Make(client_, pjrt_buffer, traceback_);
    if (sticky_) {
      TF_CHECK_OK(py_buffer.buf()->set_sticky_device(pjrt_buffer->device()));
    }
    return py_buffer;
  }
#endif

  PrimitiveType dtype() const {
#ifdef JAX_ENABLE_IFRT
    return *ifrt::ToPrimitiveType(ifrt_array_->dtype());
#else
    return buffers_.at(0)->on_device_shape().element_type();
#endif
  }

#ifdef JAX_ENABLE_IFRT
  ifrt::Array* ifrt_array() const { return ifrt_array_.get(); }

  // Short-term escape hatch to get PjRtBuffer from PyShardedBuffer.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  PjRtBuffer* pjrt_buffer(int device_id) const {
    auto* arr = llvm::dyn_cast_or_null<ifrt::PjRtArray>(ifrt_array_.get());
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return arr->pjrt_buffers().at(device_id).get();
  }
#else
  PjRtBuffer* pjrt_buffer(int device_id) const {
    return buffers_.at(device_id).get();
  }
#endif

#ifdef JAX_ENABLE_IFRT
  int num_devices() const { return ifrt_array_->sharding().devices().size(); }
#else
  int num_devices() const { return buffers_.size(); }
#endif

  const std::shared_ptr<Traceback>& traceback() const { return traceback_; }

  Status BlockHostUntilReady();

  void Delete() {
#ifdef JAX_ENABLE_IFRT
    ifrt_array_->Delete();
#else
    for (auto& pjrt_buffer : buffers_) {
      pjrt_buffer->Delete();
    }
#endif
  }

 private:
  void Link() {
    if (!client_) return;

    CHECK(PyGILState_Check());
    next_ = client_->sharded_buffers_;
    client_->sharded_buffers_ = this;
    if (next_) {
      next_->prev_ = this;
    }
    prev_ = nullptr;
  }

  void Unlink() {
    if (!client_) return;

    CHECK(PyGILState_Check());
    if (client_->sharded_buffers_ == this) {
      client_->sharded_buffers_ = next_;
    }
    if (prev_) {
      prev_->next_ = next_;
    }
    if (next_) {
      next_->prev_ = prev_;
    }
  }

  friend class PyClient;

  std::shared_ptr<PyClient> client_;
#ifdef JAX_ENABLE_IFRT
  std::unique_ptr<ifrt::Array> ifrt_array_;
#else
  std::vector<std::shared_ptr<PjRtBuffer> > buffers_;
#endif
  std::shared_ptr<Traceback> traceback_;
  bool sticky_ = false;

  PyShardedBuffer* next_ = nullptr;
  PyShardedBuffer* prev_ = nullptr;
};

#ifdef JAX_ENABLE_IFRT
// TODO(hyeontaek): Move the following functions to a separate file.
StatusOr<ifrt::DType> ToIfRtDType(pybind11::dtype dtype);
StatusOr<pybind11::dtype> ToPybind11DType(ifrt::DType dtype);
#endif

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
