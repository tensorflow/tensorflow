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
#include <stdexcept>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// As we are deploying both a C++ and a Python implementation for DeviceArray,
// we use an empty base-class to ensure `isinstance(x, DeviceArray)` works.
//         DeviceArrayBase == DeviceArray
//              /  \
//             /    \
//    PyBuffer      _DeviceArray (Python)
//      in C++
class DeviceArrayBase {
 public:
  DeviceArrayBase() = default;
};

// Python wrapper around PjRtBuffer. We use a wrapper class:
// a) to keep the PjRtClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
//
// A `PyBuffer` can be used from Python without being wrapped in a Python
// `DeviceArray` object, at the condition there is no associated LazyExpr.
class PyBuffer : public DeviceArrayBase {
 public:
  PyBuffer(std::shared_ptr<PyClient> client, std::shared_ptr<PjRtBuffer> buffer,
           std::shared_ptr<Traceback> traceback);
  ~PyBuffer();

  std::shared_ptr<PyClient> client() const { return client_; }
  PjRtBuffer* buffer() const { return buffer_.get(); }

  ClientAndPtr<PjRtDevice> device() const;
  absl::string_view platform_name() const {
    return buffer_->client()->platform_name();
  }
  bool is_deleted() const { return buffer_->IsDeleted(); }

  StatusOr<std::unique_ptr<PyBuffer>> CopyToDevice(
      const ClientAndPtr<PjRtDevice>& dst_device) const;

  int64 OnDeviceSizeInBytes() { return buffer_->OnDeviceSizeInBytes(); }

  void Delete() {
    buffer_->Delete();
    host_value_ = nullptr;
  }

  // Makes a copy of this PyBuffer object that shares the underlying PjRtBuffer.
  // This is useful because we may wish to change JAX metadata (e.g., the sticky
  // device) without copying the buffer.
  std::unique_ptr<PyBuffer> Clone() const;

  // Returns xla::InvalidArgument if the buffer has been deleted.
  Status BlockHostUntilReady();
  Status CopyToHostAsync();

  const Shape& shape() { return buffer_->on_device_shape(); }

  StatusOr<std::uintptr_t> UnsafeBufferPointer() const;

  // Implementation of the CUDA array interface for sharing GPU buffers with
  // other Python libraries.
  StatusOr<pybind11::dict> CudaArrayInterface();

  // PEP 3118 Python buffer protocol implementation.
  static PyBufferProcs* BufferProtocol();

  Traceback* traceback() { return traceback_.get(); }

  // Returns the size (i.e. number of elements) of the (host) numpy array.
  StatusOr<int64> size();

  // Returns the number of dimensions of the (host) numpy array.
  int ndim() const { return buffer()->on_device_shape().dimensions_size(); }

  pybind11::tuple python_shape() const;
  pybind11::dtype python_dtype() const;

  // Representing the logical view of the underlying dynamic shapes.
  StatusOr<Shape> xla_dynamic_shape();

  void SetStickyDevice(pybind11::object sticky_device);
  pybind11::object GetStickyDevice() const { return sticky_device_.value(); }

  StatusOr<pybind11::object> AsNumPyArray(pybind11::handle this_obj);

  void SetAval(pybind11::object aval);
  pybind11::object GetAval() const { return aval_.value(); }

 private:
  friend class PyClient;

  struct HostValue {
    absl::Notification ready;
    Status status;
    std::shared_ptr<xla::Literal> value;
  };
  std::shared_ptr<PyClient> client_;
  std::shared_ptr<PjRtBuffer> buffer_;
  std::shared_ptr<Traceback> traceback_;
  std::shared_ptr<HostValue> host_value_;  // Protected by the GIL.

  absl::optional<pybind11::object> sticky_device_ = absl::nullopt;
  // TODO(jblespiau): It's currently there for convenience but maybe we can do
  // without it (adding `weak_type` instead).
  absl::optional<pybind11::object> aval_ = absl::nullopt;
  absl::optional<Shape> dynamic_shape_ = absl::nullopt;
  // Doubly-linked list of all PyBuffers known to the client. Protected by the
  // GIL. Since multiple PyBuffers may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyBuffer* next_;
  PyBuffer* prev_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_BUFFER_H_
