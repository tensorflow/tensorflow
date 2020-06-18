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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

// Python wrapper around PjRtExecutable. We use a wrapper class:
// a) to keep the PyClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
class PyExecutable {
 public:
  PyExecutable(std::shared_ptr<PyClient> client,
               std::unique_ptr<PjRtExecutable> executable,
               std::shared_ptr<Traceback> traceback);
  ~PyExecutable();

  std::shared_ptr<PyClient> client() const { return client_; }

  const std::vector<std::pair<int, int>>& local_logical_device_ids() const {
    return executable_->local_logical_device_ids();
  }

  std::vector<ClientAndPtr<Device>> LocalDevices() const;

  int64 SizeOfGeneratedCodeInBytes() const {
    return executable_->SizeOfGeneratedCodeInBytes();
  }

  void Delete() { return executable_->Delete(); }

  StatusOr<std::vector<std::unique_ptr<PyBuffer>>> Execute(
      absl::Span<PyBuffer* const> args);

  StatusOr<std::vector<std::vector<std::unique_ptr<PyBuffer>>>>
  ExecuteOnLocalDevices(absl::Span<const std::vector<PyBuffer*>> args);

  StatusOr<std::vector<std::shared_ptr<HloModule>>> HloModules() const;

  Traceback* traceback() { return traceback_.get(); }

 private:
  friend class PyClient;

  std::shared_ptr<PyClient> client_;
  std::unique_ptr<PjRtExecutable> executable_;
  std::shared_ptr<Traceback> traceback_;

  // Doubly-linked list of all executables known to the client. Protected by the
  // GIL.
  PyExecutable* next_;
  PyExecutable* prev_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
