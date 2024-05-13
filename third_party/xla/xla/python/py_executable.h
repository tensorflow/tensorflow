/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_PY_EXECUTABLE_H_
#define XLA_PYTHON_PY_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/exceptions.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/pjrt_ifrt/pjrt_executable.h"
#include "xla/python/py_array.h"
#include "xla/python/py_client.h"
#include "xla/python/traceback.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace xla {

class PyToken {
 public:
  PyToken() = default;
  explicit PyToken(PjRtFuture<> future) : future_(std::move(future)) {}

  static PyToken ReadyPyToken() {
    return PyToken(PjRtFuture<>(absl::OkStatus()));
  }

  absl::Status Await();

 private:
  PjRtFuture<> future_;
};

// PyShardedToken contains a PyToken for each device's execution.
class PyShardedToken {
 public:
  // Default construction creates a always-ready token.
  PyShardedToken() = default;
  explicit PyShardedToken(std::vector<PjRtFuture<>> futures)
      : futures_(std::move(futures)) {}

  PyToken GetPyToken(int device_id) const {
    if (futures_.empty()) return PyToken::ReadyPyToken();
    return PyToken(futures_.at(device_id));
  }

  absl::Status Await();

 private:
  std::vector<PjRtFuture<>> futures_;
};

class PyExecuteResults {
 public:
  PyExecuteResults(const nb_class_ptr<PyClient>& client,
                   std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays,
                   int num_computations, PyShardedToken token,
                   PjRtFuture<> result_status = PjRtFuture<>());

  std::vector<std::vector<PyArray>> DisassembleIntoSingleDeviceArrays();

  std::vector<std::vector<PyArray>> DisassemblePrefixIntoSingleDeviceArrays(
      size_t n);

  std::vector<nanobind::object> ConsumeWithHandlers(
      std::vector<std::variant<const PyArrayResultHandler*, nanobind::object>>
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
  nb_class_ptr<PyClient> client_;
  std::vector<tsl::RCReference<ifrt::Array>> ifrt_arrays_;
  int num_computations_;
  PyShardedToken token_;
  // Only set if the computation has tokens.
  PjRtFuture<> result_status_;
};

using ExecuteShardedArg = std::variant<PyArray, std::vector<PyArray>>;

// Python wrapper around PjRtExecutable. We use a wrapper class:
// a) to keep the PyClient alive via a std::shared_ptr<>
// b) to add Python-specific functionality.
class PyLoadedExecutable {
 public:
  PyLoadedExecutable(
      nb_class_ptr<PyClient> client,
      std::shared_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable,
      std::optional<nb_traceback> traceback,
      std::optional<std::string> fingerprint);
  ~PyLoadedExecutable();

  nb_class_ptr<PyClient> client() const { return client_; }
  ifrt::LoadedExecutable* ifrt_loaded_executable() const {
    return ifrt_loaded_executable_.get();
  }

  std::shared_ptr<ifrt::LoadedExecutable> shared_ifrt_loaded_executable() {
    return ifrt_loaded_executable_;
  }

  absl::Span<const PjRtLoadedExecutable::LogicalDeviceIds>
  addressable_device_logical_ids() const {
    return ifrt_loaded_executable_->addressable_device_logical_ids();
  }

  std::vector<nb_class_ptr<PyDevice>> AddressableDevices() const;

  int64_t SizeOfGeneratedCodeInBytes() const {
    return ifrt_loaded_executable_->SizeOfGeneratedCodeInBytes();
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const {
    nanobind::gil_scoped_release scope;
    return ifrt_loaded_executable_->GetCompiledMemoryStats();
  }

  absl::StatusOr<absl::flat_hash_map<std::string, PjRtValueType>>
  GetCostAnalysis() const {
    return ifrt_loaded_executable_->GetCostAnalysis();
  }

  void Delete() {
    // TODO(hyeontaek): Return absl::Status.
    TF_CHECK_OK(ifrt_loaded_executable_->Delete().Await());
  }

  bool is_deleted() { return ifrt_loaded_executable_->IsDeleted(); }

  // Takes args indexed by argid then deviceid, transposes them, and passes to
  // PjRtExecutable::Execute. The result is similarly transposed back into the
  // argid,deviceid format.
  // args is [num_args x num_devices].
  absl::StatusOr<std::vector<std::vector<PyArray>>>
  ExecuteShardedOnLocalDevices(absl::Span<const ExecuteShardedArg> args);

  absl::StatusOr<std::pair<std::vector<std::vector<PyArray>>, PyShardedToken>>
  ExecuteShardedOnLocalDevicesWithTokens(
      absl::Span<const ExecuteShardedArg> args);

  absl::StatusOr<PyExecuteResults> ExecuteSharded(
      std::vector<ExecuteShardedArg> args, bool with_tokens);

  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> HloModules() const;

  absl::StatusOr<std::vector<std::vector<std::string_view>>>
  GetOutputMemoryKinds() const;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>> GetParameterLayouts()
      const;

  absl::StatusOr<std::vector<std::unique_ptr<PjRtLayout>>> GetOutputLayouts()
      const;

  std::optional<std::vector<OpSharding>> GetParameterShardings() const;

  std::optional<std::vector<OpSharding>> GetOutputShardings() const;

  const std::optional<nb_traceback>& traceback() { return traceback_; }

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
  void KeepAlive(nanobind::object obj);

 private:
  friend class PyClient;

  nb_class_ptr<PyClient> client_;
  std::shared_ptr<ifrt::LoadedExecutable> ifrt_loaded_executable_;
  std::optional<nb_traceback> traceback_;

  // Identical executables (i.e. representing the same program) will have the
  // same fingerprint. nullopt on platforms or executables where fingerprints
  // aren't implemented.
  std::optional<std::string> fingerprint_;

  // The options to pass to `executable_.Execute`.
  ExecuteOptions options_;

  // Python objects to keep alive as requested by user.
  std::vector<nanobind::object> keepalives_;

  // Doubly-linked list of all executables known to the client. Protected by the
  // GIL.
  PyLoadedExecutable* next_;
  PyLoadedExecutable* prev_;
};

}  // namespace xla

#endif  // XLA_PYTHON_PY_EXECUTABLE_H_
