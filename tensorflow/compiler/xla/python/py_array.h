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
#include <utility>
#include <vector>

#include "pybind11/pybind11.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/types.h"

namespace xla {

// The C++ implementation of jax.Array. A few key methods and data members are
// implemented in C++ for performance, while most of the functionalities are
// still implemented in python.
//
// TODO(chky): Consider replacing the usage of PyShardedBuffer with PyArray as
// PyArray is more general.
class PyArray : public pybind11::object {
 public:
  PYBIND11_OBJECT(PyArray, pybind11::object, PyArray::IsPyArray);

  // "__init__" methods. Only used in python
  static void PyInit(pybind11::object self, pybind11::object aval,
                     pybind11::object sharding,
                     absl::Span<const PyBuffer::object> py_buffers,
                     bool committed, bool skip_checks);

  static void PyInit(pybind11::object self, pybind11::object aval,
                     pybind11::object sharding,
                     absl::Span<const PyArray> py_arrays, bool committed,
                     bool skip_checks);

  // Only used in C++
  PyArray(pybind11::object aval, bool weak_type, pybind11::dtype dtype,
          std::vector<int64_t> shape, pybind11::object sharding,
          std::shared_ptr<PyClient> py_client,
          std::shared_ptr<Traceback> traceback,
          std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers, bool committed,
          bool skip_checks = true);

  static Status RegisterTypes(pybind11::module& m);

  struct Storage {
    Storage(pybind11::object aval, bool weak_type, pybind11::dtype dtype,
            std::vector<int64_t> shape, pybind11::object sharding,
            bool committed, std::shared_ptr<PyClient> py_client,
            std::shared_ptr<Traceback> traceback,
            std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers)
        : aval(std::move(aval)),
          weak_type(weak_type),
          dtype(std::move(dtype)),
          shape(std::move(shape)),
          sharding(std::move(sharding)),
          committed(committed),
          py_client(std::move(py_client)),
          traceback(std::move(traceback)),
          pjrt_buffers(std::move(pjrt_buffers)) {}

    pybind11::object aval;
    bool weak_type = false;
    pybind11::dtype dtype;
    std::vector<int64_t> shape;

    pybind11::object sharding;
    pybind11::object npy_value = pybind11::none();
    bool committed = false;

    std::shared_ptr<PyClient> py_client;
    std::shared_ptr<Traceback> traceback;
    std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers;

    // optional field, used only in python
    std::vector<PyBuffer::object> py_buffers;
  };

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

  std::vector<std::shared_ptr<PjRtBuffer>>& pjrt_buffers() {
    return GetStorage().pjrt_buffers;
  }
  const std::vector<std::shared_ptr<PjRtBuffer>>& pjrt_buffers() const {
    return GetStorage().pjrt_buffers;
  }
  std::vector<PyBuffer::object>& py_buffers() {
    return GetStorage().py_buffers;
  }
  const std::vector<PyBuffer::object>& py_buffers() const {
    return GetStorage().py_buffers;
  }

  pybind11::object arrays();
  Status set_arrays(pybind11::object obj);

  PjRtBuffer* GetBuffer(int device_id) const {
    return pjrt_buffers().at(device_id).get();
  }

  const std::shared_ptr<PjRtBuffer>& GetSharedPtrBuffer(int device_id) const {
    return pjrt_buffers().at(device_id);
  }

  int num_shards() const { return pjrt_buffers().size(); }

  static pybind11::handle type() {
    DCHECK(type_);
    return pybind11::handle(type_);
  }

  static bool IsPyArray(pybind11::handle arg) {
    return arg.get_type().is(PyArray::type());
  }

  Status BlockUntilReady() const;

 private:
  void CheckAndRearrange();

  Storage& GetStorage();
  const Storage& GetStorage() const;

  static Status SetUpType();

  inline static PyObject* type_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_ARRAY_H_
