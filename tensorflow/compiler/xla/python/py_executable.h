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
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_executable.h"
#include "tensorflow/compiler/xla/python/py_array.h"
#include "tensorflow/compiler/xla/python/py_buffer.h"
#include "tensorflow/compiler/xla/python/py_client.h"
#include "tensorflow/compiler/xla/python/traceback.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class PyToken {
 public:
  PyToken() = default;
  explicit PyToken(PjRtFuture<Status> future) : future_(std::move(future)) {}

  static PyToken ReadyPyToken() {
    return PyToken(PjRtFuture<Status>(OkStatus()));
  }

  Status Await();

 private:
  PjRtFuture<Status> future_;
};

// PyShardedToken contains a PyToken for each device's execution.
class PyShardedToken {
 public:
  // Default construction creates a always-ready token.
  PyShardedToken() = default;
  explicit PyShardedToken(std::vector<PjRtFuture<Status>> futures)
      : futures_(std::move(futures)) {}

  PyToken GetPyToken(int device_id) const {
    if (futures_.empty()) return PyToken::ReadyPyToken();
    return PyToken(futures_.at(device_id));
  }

  Status Await();

 private:
  std::vector<PjRtFuture<Status>> futures_;
};

class PyExecuteResults {
 public:
  PyExecuteResults(const std::shared_ptr<PyClient>& client,
                   std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
                   int num_computations, PyShardedToken token);

  std::vector<std::vector<PyArray>> DisassembleIntoSingleDeviceArrays();

  std::vector<std::vector<PyArray>> DisassemblePrefixIntoSingleDeviceArrays(
      size_t n);

  std::vector<pybind11::object> ConsumeWithHandlers(
      std::vector<std::variant<const PyArrayResultHandler*, pybind11::object>>
          out_handlers);

  std::vector<tsl::RCReference<ifrt::Array>> Consume();

  PyShardedToken ConsumeToken();

  size_t Size() const {
    CheckNotDisassembled();
    return ifrt_arrays_.size();
  }

  void CheckNotDisassembled() const;

 private:
  bool is_exploded_ = false;
  bool token_consumed_ = false;
  std::shared_ptr<PyClient> client_;
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays_;
  int num_computations_;
  PyShardedToken token_;
};

using ExecuteShardedArg = std::variant<PyArray, std::vector<PyArray>>;

// Python wrapper around PjRtExecutable. We use a wrapper class:
// a) to keep the PyClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
class PyLoadedExecutable
    : public std::enable_shared_from_this<PyLoadedExecutable> {
 public:
  PyLoadedExecutable(
      std::shared_ptr<PyClient> client,
      std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable,
      std::shared_ptr<Traceback> traceback,
      std::optional<std::string> fingerprint,
      std::vector<pybind11::capsule> host_callbacks);
  ~PyLoadedExecutable();

  std::shared_ptr<PyClient> client() const { return client_; }
  ifrt::LoadedExecutable* ifrt_loaded_executable() const {
    return ifrt_loaded_executable_.get();
  }

  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const {
    return ifrt_loaded_executable_->addressable_device_logical_ids();
  }

  std::vector<ClientAndPtr<PjRtDevice>> AddressableDevices() const;

  int64_t SizeOfGeneratedCodeInBytes() const {
    return ifrt_loaded_executable_->SizeOfGeneratedCodeInBytes();
  }

  StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const {
    return ifrt_loaded_executable_->GetCompiledMemoryStats();
  }

  void Delete() {
    // TODO(hyeontaek): Return Status.
    TF_CHECK_OK(ifrt_loaded_executable_->Delete().Await());
  }

  bool is_deleted() { return ifrt_loaded_executable_->IsDeleted(); }

  // Takes args indexed by argid then deviceid, transposes them, and passes to
  // PjRtExecutable::Execute. The result is similarly transposed back into the
  // argid,deviceid format.
  // args is [num_args x num_devices].
  StatusOr<std::vector<std::vector<PyArray>>> ExecuteShardedOnLocalDevices(
      absl::Span<const ExecuteShardedArg> args);

  StatusOr<std::pair<std::vector<std::vector<PyArray>>, PyShardedToken>>
  ExecuteShardedOnLocalDevicesWithTokens(
      absl::Span<const ExecuteShardedArg> args);

  StatusOr<PyExecuteResults> ExecuteSharded(std::vector<ExecuteShardedArg> args,
                                            bool with_tokens);

  StatusOr<std::vector<std::shared_ptr<HloModule>>> HloModules() const;

  std::optional<std::vector<OpSharding>> GetParameterShardings() const;

  std::optional<std::vector<OpSharding>> GetOutputShardings() const;

  Traceback* traceback() { return traceback_.get(); }

  ifrt::LoadedExecutable* ifrt_executable() const {
    return ifrt_loaded_executable_.get();
  }

  // Short-term escape hatch to get PjRtLoadedExecutable from PyExecutable.
  // TODO(hyeontaek): Migrate all users of this method to be agnostic of PjRt.
  PjRtLoadedExecutable* pjrt_executable() const {
    auto* exec = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleLoadedExecutable>(
        ifrt_loaded_executable_.get());
    if (exec == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return exec->pjrt_loaded_executable();
  }
  std::shared_ptr<PjRtLoadedExecutable> shared_ptr_pjrt_executable() {
    auto* exec = llvm::dyn_cast_or_null<ifrt::PjRtCompatibleLoadedExecutable>(
        ifrt_loaded_executable_.get());
    if (exec == nullptr) {
      throw XlaRuntimeError(
          "This operation is implemented for a PjRt-compatible backend only.");
    }
    return exec->shared_ptr_pjrt_loaded_executable();
  }

  const ExecuteOptions& options() const { return options_; }
  const std::optional<std::string>& fingerprint() const { return fingerprint_; }

  // Keep `obj` alive as long as PyLoadedExecutable.
  void KeepAlive(pybind11::object obj);

 private:
  friend class PyClient;

  std::shared_ptr<PyClient> client_;
  std::unique_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable_;
  std::shared_ptr<Traceback> traceback_;

  // Identical executables (i.e. representing the same program) will have the
  // same fingerprint. nullopt on platforms or executables where fingerprints
  // aren't implemented.
  std::optional<std::string> fingerprint_;

  // The python callbacks implemented using send/recv support.
  std::vector<pybind11::capsule> host_callbacks_;

  // The options to pass to `executable_.Execute`.
  ExecuteOptions options_;

  // Python objects to keep alive as requested by user.
  std::vector<pybind11::object> keepalives_;

  // Doubly-linked list of all executables known to the client. Protected by the
  // GIL.
  PyLoadedExecutable* next_;
  PyLoadedExecutable* prev_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PY_EXECUTABLE_H_
