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

#include "tensorflow/compiler/xla/python/python_ref_manager.h"

#include "absl/container/inlined_vector.h"

namespace xla {

namespace py = pybind11;

PythonRefManager::ManagedPyObjects::ManagedPyObjects(
    PythonRefManager* manager, absl::Span<pybind11::object> objects)
    : manager_(manager) {
  objects_.reserve(objects.size());
  for (pybind11::object& object : objects) {
    objects_.push_back(std::move(object));
  }
}

PythonRefManager::ManagedPyObjects::~ManagedPyObjects() {
  if (manager_ && !objects_.empty()) {
    manager_->AddGarbage(absl::MakeSpan(objects_));
  }
}

std::shared_ptr<PythonRefManager::ManagedPyObjects>
PythonRefManager::ManageReference(py::object object) {
  return std::make_shared<ManagedPyObjects>(this,
                                            absl::Span<py::object>(&object, 1));
}

std::shared_ptr<PythonRefManager::ManagedPyObjects>
PythonRefManager::ManageReferences(absl::Span<py::object> objects) {
  return std::make_shared<ManagedPyObjects>(this, objects);
}

void PythonRefManager::AddGarbage(absl::Span<py::object> garbage) {
  absl::MutexLock lock(&mu_);
  // We want to collect arbitrary python garbage (e.g., buffers) aggressively.
  garbage_count_.fetch_add(100, std::memory_order_relaxed);
  for (py::object& o : garbage) {
    python_garbage_.push_back(std::move(o));
  }
}

void PythonRefManager::AddGarbage(
    absl::Span<std::pair<PyCodeObject*, int> const> garbage) {
  absl::MutexLock lock(&mu_);
  // We don't care about collecting stack frame objects often. We grab a lot of
  // tracebacks and the code objects are most likely live for the entire
  // process.
  garbage_count_.fetch_add(1, std::memory_order_relaxed);
  for (const auto& o : garbage) {
    python_garbage_.push_back(py::reinterpret_steal<py::object>(
        reinterpret_cast<PyObject*>(o.first)));
  }
}

void PythonRefManager::CollectGarbage() {
  // TODO(phawkins): we should CHECK(PyGILState_Check());
  std::deque<pybind11::object> garbage;
  {
    absl::MutexLock lock(&mu_);
    garbage_count_ = 0;
    garbage.swap(python_garbage_);
  }
  // We defer deleting garbage until the lock is released. It's possible that
  // deleting garbage will lead to more Python garbage being added; if we held
  // the lock we would deadlock because absl::Mutex is not reentrant.
}

PythonRefManager* GlobalPyRefManager() {
  static PythonRefManager* static_ref_manager = new PythonRefManager();
  return static_ref_manager;
}

}  // namespace xla
