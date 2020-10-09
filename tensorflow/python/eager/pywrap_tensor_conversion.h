/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
#define TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_

// Place `<locale>` before <Python.h> to avoid build failure in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

// Wrapper-class allowing to use Python hashing/comparison functions
// for PyObject*.
//
// Note that unlike Safe_PyObjectPtr this class does not steal a
// reference to a Python object. The caller is responsible for doing
// Py_INCREF/Py_DECREF.
struct PyObjectPtr {
  template <typename H>
  friend H AbslHashValue(H h, const PyObjectPtr& obj) {
    return H::combine(std::move(h), PyObject_Hash(obj.ptr));
  }

  explicit PyObjectPtr(PyObject* ptr) : ptr(ptr) {}

  explicit inline operator PyObject*() const { return ptr; }

  inline bool operator==(const PyObjectPtr& other) const {
    // We require exact type equality to account for 0 == 0.0 == False.
    if (Py_TYPE(ptr) != Py_TYPE(other.ptr)) {
      return false;
    }

    bool result = PyObject_RichCompareBool(ptr, other.ptr, Py_EQ) > 0;
    CHECK(!PyErr_Occurred());
    return result;
  }

 private:
  PyObject* ptr;
};

// Cache mapping PyObject* to the corresponding on-device TFE_TensorHandles.
// Used to speed up ConvertToEagerTensor for scalars.
// TODO(slebedev): move ConvertToEagerTensor here.
struct TFE_TensorHandleCache {
  static TFE_TensorHandleCache* Get();

  TFE_TensorHandleCache() { cache.reserve(64); }
  ~TFE_TensorHandleCache() { DecrefUnrefAll(); }

  TFE_TensorHandle* Lookup(PyObject* value, tensorflow::DataType dtype,
                           TFE_Context* ctx,
                           absl::string_view device_name) const;

  void Insert(PyObject* value, tensorflow::DataType dtype, TFE_Context* ctx,
              absl::string_view device_name, TFE_TensorHandle* h);

  void Clear();

 private:
  // TODO(kkb): Instead of `TFE_Context*` key, ideally Python's context object
  // should have TFE_TensorHandleCache instance. Migrate once we Python context
  // object is backed by C++ data structure. b/169790439
  using Key = std::tuple<PyObjectPtr, tensorflow::DataType, TFE_Context*,
                         absl::string_view>;

  void DecrefUnrefAll() {
    for (const auto& p : cache) {
      Py_DECREF(static_cast<PyObject*>(std::get<0>(p.first)));
      TFE_DeleteTensorHandle(p.second);
    }
  }

  // Not guarded by a mutex because the code is only used while the
  // GIL is held.
  absl::flat_hash_map<Key, TFE_TensorHandle*> cache;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
