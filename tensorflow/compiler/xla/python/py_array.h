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
class PyArray {
 public:
  static void RegisterTypes(pybind11::module& m);

  // Only used in python
  PyArray(pybind11::object aval, pybind11::object sharding,
          absl::Span<const PyBuffer::object> py_buffers, bool committed,
          bool skip_checks, pybind11::object fast_path_args);

  PyArray(pybind11::object aval, pybind11::object sharding,
          absl::Span<const PyArray* const> py_arrays, bool committed,
          bool skip_checks, pybind11::object fast_path_args);

  // Only used in C++
  PyArray(pybind11::object aval, bool weak_type, PrimitiveType dtype,
          std::vector<int64_t> shape, pybind11::object sharding,
          std::shared_ptr<PyClient> py_client,
          std::shared_ptr<Traceback> traceback,
          std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers, bool committed,
          bool skip_checks, pybind11::object fast_path_args = pybind11::none());

  const pybind11::object& aval() const { return aval_; }
  void set_aval(pybind11::object aval) { aval_ = std::move(aval); }

  const pybind11::object& sharding() const { return sharding_; }

  pybind11::object arrays() const;
  Status set_arrays(pybind11::object obj);

  bool committed() const { return committed_; }

  const pybind11::object& fast_path_args() const { return fast_path_args_; }

  const pybind11::object& npy_value() const { return npy_value_; }
  void set_npy_value(pybind11::object v) { npy_value_ = std::move(v); }

  const std::shared_ptr<PyClient>& py_client() const { return py_client_; }

  PjRtBuffer* GetBuffer(int device_id) const {
    return pjrt_buffers_.at(device_id).get();
  }

  const std::shared_ptr<PjRtBuffer>& GetSharedPtrBuffer(int device_id) const {
    return pjrt_buffers_.at(device_id);
  }

  int num_shards() const { return pjrt_buffers_.size(); }

  static pybind11::handle type() {
    static pybind11::handle type = pybind11::type::handle_of<PyArray>();
    return type;
  }

  bool weak_type() const { return weak_type_; }
  PrimitiveType dtype() const { return dtype_; }
  absl::Span<const int64_t> shape() const { return shape_; }

  Status BlockUntilReady() const {
    pybind11::gil_scoped_release gil_release;
    Status status;
    for (const auto& pjrt_buffer : pjrt_buffers_) {
      auto s = pjrt_buffer->GetReadyFuture().Await();
      if (!s.ok()) status = std::move(s);
    }
    return status;
  }

 private:
  void Check();
  void Rearrange();

  pybind11::object aval_;
  bool weak_type_ = false;
  PrimitiveType dtype_;
  std::vector<int64_t> shape_;

  pybind11::object sharding_;
  pybind11::object fast_path_args_ = pybind11::none();
  pybind11::object npy_value_ = pybind11::none();
  bool committed_ = false;

  std::shared_ptr<PyClient> py_client_;
  std::shared_ptr<Traceback> traceback_;
  std::vector<std::shared_ptr<PjRtBuffer>> pjrt_buffers_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_ARRAY_H_
