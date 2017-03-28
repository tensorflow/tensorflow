/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Must be included first.
#include "tensorflow/python/lib/core/numpy.h"

#include <vector>

#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/python/lib/core/ndarray_tensor_bridge.h"

namespace tensorflow {

// Mutex used to serialize accesses to cached vector of pointers to python
// arrays to be dereferenced.
static mutex* DelayedDecrefLock() {
  static mutex* decref_lock = new mutex;
  return decref_lock;
}

// Caches pointers to numpy arrays which need to be dereferenced.
static std::vector<void*>* DecrefCache() {
  static std::vector<void*>* decref_cache = new std::vector<void*>;
  return decref_cache;
}

// Destructor passed to TF_NewTensor when it reuses a numpy buffer. Stores a
// pointer to the pyobj in a buffer to be dereferenced later when we're actually
// holding the GIL.
void DelayedNumpyDecref(void* data, size_t len, void* obj) {
  mutex_lock ml(*DelayedDecrefLock());
  DecrefCache()->push_back(obj);
}

// Actually dereferences cached numpy arrays. REQUIRES being called while
// holding the GIL.
void ClearDecrefCache() {
  mutex_lock ml(*DelayedDecrefLock());
  for (void* obj : *DecrefCache()) {
    Py_DECREF(reinterpret_cast<PyObject*>(obj));
  }
  DecrefCache()->clear();
}

}  // namespace tensorflow
