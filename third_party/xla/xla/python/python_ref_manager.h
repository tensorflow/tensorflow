/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PYTHON_REF_MANAGER_H_
#define XLA_PYTHON_PYTHON_REF_MANAGER_H_

#include <Python.h>

#include <atomic>
#include <deque>
#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "nanobind/nanobind.h"

namespace xla {

// Class that manages destruction of Python objects.
//
// We must not destroy Python objects without holding the GIL. However, we
// frequently want to hold references to Python objects for the duration of
// an asynchronous transfer on a Stream, and release our reference when the
// transfer completes.
//
// This class holds references to Python objects outside a GIL scope, that can
// be collected later when the GIL is held by calling CollectGarbage().
class PythonRefManager {
 public:
  PythonRefManager() = default;

  // Holds references to a set of nanobind::objects, adding the references to
  // the PythonRefManager on destruction.
  class ManagedPyObjects {
   public:
    ManagedPyObjects() = default;
    ManagedPyObjects(PythonRefManager* manager,
                     absl::Span<nanobind::object> objects);

    ~ManagedPyObjects();

    ManagedPyObjects(const ManagedPyObjects& other) = delete;
    ManagedPyObjects(ManagedPyObjects&& other) = default;
    ManagedPyObjects& operator=(const ManagedPyObjects& other) = delete;
    ManagedPyObjects& operator=(ManagedPyObjects&& other) noexcept = default;

   private:
    PythonRefManager* manager_ = nullptr;
    absl::InlinedVector<nanobind::object, 1> objects_;
  };

  // Creates a managed std::shared_ptr to an object. When the shared_ptr is
  // destroyed, the reference to 'object' will be added to python_garbage_,
  // and collected next time CollectGarbage() is called.
  std::shared_ptr<ManagedPyObjects> ManageReference(nanobind::object object);
  std::shared_ptr<ManagedPyObjects> ManageReferences(
      absl::Span<nanobind::object> objects);

  // Adds garbage objects to the manager.
  void AddGarbage(nanobind::object garbage);
  void AddGarbage(absl::Span<nanobind::object> garbage);
  void AddGarbage(absl::Span<std::pair<PyCodeObject*, int> const> garbage);

  // Releases the contents of python_garbage_. Requires that the GIL is held.
  // The client calls this method during API entry points where the GIL is held
  // to free any garbage that has accumulated.
  void CollectGarbage();

  // Cheaper version of CollectGarbage() with relaxed consistency and frequency.
  // The purpose of this function is to amortize lock acquisition costs over
  // a larger number of API calls.
  void MaybeCollectGarbage() {
    if (garbage_count_.load(std::memory_order_relaxed) >= 100) {
      CollectGarbage();
    }
  }

 private:
  absl::Mutex mu_;
  std::deque<nanobind::object> python_garbage_ ABSL_GUARDED_BY(mu_);

  // Writes to garbage_count_ are protected by mu_, reads are not protected.
  std::atomic<int> garbage_count_{0};
};

// A global PythonRefManager. Unless `CollectGarbage()` is called before
// shutdown, this container will hold on to Python objects and thus cause a
// leak. This behavior is similar to `tensorflow::ClearDecRefCache()`.
PythonRefManager* GlobalPyRefManager();

}  // namespace xla

#endif  // XLA_PYTHON_PYTHON_REF_MANAGER_H_
