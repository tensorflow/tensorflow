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
  if (manager_) {
    absl::MutexLock lock(&manager_->mu_);
    for (pybind11::object& object : objects_) {
      manager_->python_garbage_.push_back(std::move(object));
    }
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
  for (py::object& o : garbage) {
    python_garbage_.push_back(std::move(o));
  }
}

void PythonRefManager::CollectGarbage() {
  // TODO(phawkins): we should CHECK(PyGILState_Check());
  std::deque<pybind11::object> garbage;
  {
    absl::MutexLock lock(&mu_);
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
