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
#include <string>
#include <tuple>
#include <utility>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/core/common_runtime/eager/tensor_handle.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// Wrapper-class allowing to use Python hashing/comparison functions
// for PyObject*.
//
// Note that unlike Safe_PyObjectPtr this class does not steal a
// reference to a Python object. The caller is responsible for doing
// Py_INCREF/Py_DECREF.
class PyObjectPtr {
 public:
  template <typename H>
  friend H AbslHashValue(H h, const PyObjectPtr& obj) {
    Py_hash_t hash = PyObject_Hash(obj.ptr_);
    CHECK_NE(hash, -1);        // Crash OK
    CHECK(!PyErr_Occurred());  // Crash OK
    return H::combine(std::move(h), hash);
  }

  explicit PyObjectPtr(PyObject* ptr) : ptr_(ptr) {}

  explicit operator PyObject*() const { return ptr_; }

  bool operator==(const PyObjectPtr& other) const {
    // We require exact type equality to account for 0 == 0.0 == False.
    if (Py_TYPE(ptr_) != Py_TYPE(other.ptr_)) {
      return false;
    }

    bool result = PyObject_RichCompareBool(ptr_, other.ptr_, Py_EQ) > 0;
    CHECK(!PyErr_Occurred());  // Crash OK
    return result;
  }

 private:
  PyObject* ptr_;
};

// Cache mapping PyObject* to the corresponding on-device TFE_TensorHandles.
// Used to speed up ConvertToEagerTensor for scalars.
// TODO: b/169790439 - move ConvertToEagerTensor here.
class TFE_TensorHandleCache {
 public:
  static TFE_TensorHandleCache* Get();

  TFE_TensorHandleCache() { cache_.reserve(64); }

  TFE_TensorHandleCache(const TFE_TensorHandleCache&) = delete;
  TFE_TensorHandleCache& operator=(const TFE_TensorHandleCache&) = delete;

  ~TFE_TensorHandleCache() { DecrefUnrefAll(); }

  TFE_TensorHandle* Lookup(PyObject* value, DataType dtype, TFE_Context* ctx,
                           absl::string_view device_name) const;

  void Insert(PyObject* value, DataType dtype, TFE_Context* ctx,
              absl::string_view device_name, TFE_TensorHandle* h);

  void Clear();

  // Maximum number of entries before the cache is cleared. Prevents unbounded
  // growth when many distinct scalar values are created in a loop.
  static constexpr size_t kMaxCacheSize = 1024;

 private:
  // TODO: b/169790439 - Instead of `TFE_Context*` key, ideally Python's context
  // object should have TFE_TensorHandleCache instance. Migrate once Python
  // context object is backed by C++ data structure.
  using Key = std::tuple<PyObjectPtr, DataType, TFE_Context*, std::string>;
  using LookupKey =
      std::tuple<PyObjectPtr, DataType, TFE_Context*, absl::string_view>;

  struct KeyHash {
    using is_transparent = void;

    template <typename Tuple>
    size_t operator()(const Tuple& t) const {
      return absl::Hash<Tuple>{}(t);
    }
  };

  struct KeyEqual {
    using is_transparent = void;

    template <typename Tuple1, typename Tuple2>
    bool operator()(const Tuple1& lhs, const Tuple2& rhs) const {
      return lhs == rhs;
    }
  };

  using Cache = absl::flat_hash_map<Key, TFE_TensorHandle*, KeyHash, KeyEqual>;

  Cache ExtractCache() {
#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(mu_);
#endif  // Py_GIL_DISABLED
    Cache temp_cache = std::move(cache_);
    cache_.clear();
    return temp_cache;
  }

  static void DecrefUnrefAll(Cache temp_cache) {
    for (const auto& [key, value] : temp_cache) {
      Py_DECREF(static_cast<PyObject*>(std::get<0>(key)));
      TFE_DeleteTensorHandle(value);
    }
  }

  void DecrefUnrefAll() { DecrefUnrefAll(ExtractCache()); }

#ifdef Py_GIL_DISABLED
  mutable absl::Mutex mu_;
#endif  // Py_GIL_DISABLED

  // Under a GIL-enabled Python, guarded by the GIL. Under a no-GIL Python,
  // guarded by mu_.
  Cache cache_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_EAGER_PYWRAP_TENSOR_CONVERSION_H_
