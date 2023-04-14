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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_ARRAY_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_ARRAY_H_

#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/Support/Casting.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

// Private to PyArray, but you cannot forward declare member classes.
struct PyArray_Storage {
  PyArray_Storage(pybind11::object aval, bool weak_type, pybind11::dtype dtype,
                  std::vector<int64_t> shape, pybind11::object sharding,
                  bool committed, std::shared_ptr<PyClient> py_client,
                  std::shared_ptr<Traceback> traceback,
                  tsl::RCReference<ifrt::Array> ifrt_array);

  // TODO(yashkatariya): remove this once the transition completes.
  struct DisableFastpath {};
  explicit PyArray_Storage(DisableFastpath);

  ~PyArray_Storage();
  pybind11::handle AsHandle();

  // TODO(yashkatariya): remove this once the transition completes.
  bool fastpath_enabled;

  pybind11::object aval;
  bool weak_type = false;
  pybind11::dtype dtype;
  std::vector<int64_t> shape;

  pybind11::object sharding;
  pybind11::object npy_value = pybind11::none();
  bool committed = false;

  std::shared_ptr<PyClient> py_client;
  std::shared_ptr<Traceback> traceback;
  tsl::RCReference<ifrt::Array> ifrt_array;

  // optional field, used only in python
  std::vector<PyArray> py_arrays;
  std::shared_ptr<PyHostValue> host_value;  // Protected by the GIL.
  std::optional<Shape> dynamic_shape = std::nullopt;

  // Doubly-linked list of all PyArrays known to the client. Protected by the
  // GIL. Since multiple PyArrays may share the same PjRtBuffer, there may be
  // duplicate PjRtBuffers in this list.
  PyArray_Storage* next;
  PyArray_Storage* prev;
};

// The C++ implementation of jax.Array. A few key methods and data members are
// implemented in C++ for performance, while most of the functionalities are
// still implemented in python.
class PyArray : public pybind11::object {
 public:
  PYBIND11_OBJECT(PyArray, pybind11::object, PyArray::IsPyArray);
  PyArray() = default;

  // "__init__" methods. Only used in python
  static void PyInit(pybind11::object self, pybind11::object aval,
                     pybind11::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks);

  // TODO(yashkatariya): remove this once the transition completes.
  struct DisableFastpath {};
  static void PyInit(pybind11::object self, DisableFastpath);

  // Only used in C++
  PyArray(pybind11::object aval, bool weak_type, pybind11::dtype dtype,
          std::vector<int64_t> shape, pybind11::object sharding,
          std::shared_ptr<PyClient> py_client,
          std::shared_ptr<Traceback> traceback,
          tsl::RCReference<ifrt::Array> ifrt_array,
          bool committed, bool skip_checks = true);

  static PyArray MakeFromSingleDeviceArray(
      std::shared_ptr<PyClient> py_client, std::shared_ptr<Traceback> traceback,
      tsl::RCReference<ifrt::Array> ifrt_array, bool weak_type, bool committed);

  static Status RegisterTypes(pybind11::module& m);

  using Storage = PyArray_Storage;

  const pybind11::object& aval() const { return GetStorage().aval; }
  void set_aval(pybind11::object aval) { GetStorage().aval = std::move(aval); }

  bool weak_type() const { return GetStorage().weak_type; }

  const pybind11::dtype& dtype() const { return GetStorage().dtype; }
  absl::Span<const int64_t> shape() const { return GetStorage().shape; }

  const pybind11::object& sharding() const { return GetStorage().sharding; }

  bool committed() const { return GetStorage().committed; }

  const pybind11::object& npy_value() const { return GetStorage().npy_value; }
  void set_npy_value(pybind11::object v) {
    GetStorage().npy_value = std::move(v);
  }

  const std::shared_ptr<PyClient>& py_client() const {
    return GetStorage().py_client;
  }

  const std::shared_ptr<Traceback>& traceback() const {
    return GetStorage().traceback;
  }

  // Returns xla::InvalidArgument if the buffer has been deleted.
  // See `PjRtFuture` for the semantics of `IsReady` and `IsKnownReady`.
  StatusOr<bool> IsReady() {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr->IsDeleted()) {
      return InvalidArgument("Array has been deleted.");
    }
    return ifrt_array_ptr->GetReadyFuture().IsReady();
  }

  ifrt::Array* ifrt_array() const { return GetStorage().ifrt_array.get(); }

  // Short-term escape hatch to get PjRtBuffers from PyArray.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  absl::Span<const std::shared_ptr<PjRtBuffer>> pjrt_buffers() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return {};
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array_ptr);
    if (arr == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return arr->pjrt_buffers();
  }

  int num_addressable_shards() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return 0;
    }
    auto* arr =
        llvm::dyn_cast_or_null<ifrt::PjRtCompatibleArray>(ifrt_array_ptr);
    if (arr == nullptr) {
      // TODO(hyeontaek): Add num_addressable_shards to ifrt.
      return num_shards();
    }
    return arr->pjrt_buffers().size();
  }

  std::vector<PyArray>& py_arrays() { return GetStorage().py_arrays; }
  const std::vector<PyArray>& py_arrays() const {
    return GetStorage().py_arrays;
  }
  const std::vector<PyArray>& py_arrays_cached();

  pybind11::object arrays();
  Status set_arrays(pybind11::object obj);

  int num_shards() const {
    ifrt::Array* ifrt_array_ptr = ifrt_array();
    if (ifrt_array_ptr == nullptr) {
      return 0;
    }
    return ifrt_array_ptr->sharding().devices().size();
  }

  // TODO(yashkatariya): remove this once the transition completes.
  bool fastpath_enabled() const { return GetStorage().fastpath_enabled; }

  static pybind11::handle type() {
    DCHECK(type_);
    return pybind11::handle(type_);
  }

  static bool IsPyArray(pybind11::handle arg) {
    return arg.get_type().is(PyArray::type());
  }

  Status BlockUntilReady() const;

  StatusOr<size_t> GetOnDeviceSizeInBytes();
  StatusOr<pybind11::object> SingleDeviceArrayToNumpyArray();
  Status CopySingleDeviceArrayToHostAsync();
  StatusOr<pybind11::dict> CudaArrayInterface();
  StatusOr<std::uintptr_t> UnsafeBufferPointer();

  Status Delete();

  bool IsDeleted() const;

  PyArray Clone() const;

  StatusOr<PyArray> CopyToDeviceWithSharding(ifrt::DeviceList devices,
                                             pybind11::object dst_sharding);

  static StatusOr<PyArray> BatchedDevicePut(
      pybind11::object aval, pybind11::object sharding,
      std::vector<pybind11::object> xs,
      std::vector<ClientAndPtr<PjRtDevice>> dst_devices, bool committed,
      bool force_copy, PjRtClient::HostBufferSemantics host_buffer_semantics,
      bool jax_enable_x64);

 private:
  StatusOr<PyArray> FetchSingleShard(std::string_view api);
  StatusOr<PyArray> AssertUnsharded(std::string_view api);

  void CheckAndRearrange();

  void SetIfrtArray(tsl::RCReference<ifrt::Array> ifrt_array);

  Storage& GetStorage();
  const Storage& GetStorage() const;

  static Status SetUpType();

  inline static PyObject* type_ = nullptr;
};

class PyArrayResultHandler {
 public:
  PyArrayResultHandler(pybind11::object aval, pybind11::object sharding,
                       bool committed, bool skip_checks);

  PyArray Call(absl::Span<const PyArray> py_arrays) const;
  PyArray Call(PyArray py_array) const;

  PyArray Call(std::shared_ptr<PyClient> py_client,
               tsl::RCReference<ifrt::Array> ifrt_array) const;

 private:
  pybind11::object aval_;
  pybind11::object sharding_;
  bool weak_type_;
  bool committed_;
  bool skip_checks_;

  pybind11::object dtype_;
  std::vector<int64_t> shape_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_ARRAY_H_
