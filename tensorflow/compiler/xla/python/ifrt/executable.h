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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_EXECUTABLE_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/array.h"
#include "tensorflow/compiler/xla/python/ifrt/future.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace ifrt {

class Client;

// Wraps a computation that has been partially compiled and can be loaded.
class Executable : public llvm::RTTIExtends<Executable, llvm::RTTIRoot> {
 public:
  // Unique name for this executable.
  virtual absl::string_view name() const = 0;

  // Returns a fingerprint of this executable.
  virtual StatusOr<std::optional<std::string>> Fingerprint() const = 0;

  // Serializes this executable into a string. The compatibility of the
  // serialized executable is implementation-specific.
  virtual StatusOr<std::string> Serialize() const = 0;

  // The following APIs are taken from `xla::PjRtExecutable` for fast
  // prototyping. TODO(hyeontaek): Factor some of them out as
  // `XlaCompatibleExecutable`.
  virtual int num_devices() const = 0;
  virtual int64_t SizeOfGeneratedCodeInBytes() const = 0;
  virtual StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const = 0;

  // TODO(hyeontaek): Move the following XLA-specific methods to
  // pjrt_executable.h and put it in an `XlaCompatibleExecutable`.

  // Returns a list of parameter `OpSharding`.
  virtual std::optional<std::vector<OpSharding>> GetParameterShardings()
      const = 0;
  // Returns a list of output `OpSharding`.
  virtual std::optional<std::vector<OpSharding>> GetOutputShardings() const = 0;
  // Returns an `HloModule` (optimized) per partition.
  virtual StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const = 0;

  static char ID;  // NOLINT
};

// Wraps a computation that has been fully compiled and loaded for execution.
class LoadedExecutable
    : public llvm::RTTIExtends<LoadedExecutable, llvm::RTTIRoot> {
 public:
  virtual Client* client() const = 0;

  // Executable methods. Note that LoadedExecutable does not inherit from
  // Executable to avoid multiple inheritance in LoadedExecutable
  // implementations.

  // Unique name for this executable.
  virtual absl::string_view name() const = 0;

  // Returns a fingerprint of this executable.
  virtual StatusOr<std::optional<std::string>> Fingerprint() const = 0;

  // Serializes this executable into a string. The compatibility of the
  // serialized executable is implementation-specific.
  virtual StatusOr<std::string> Serialize() const = 0;

  // The following APIs are taken from `xla::PjRtExecutable` for fast
  // prototyping.

  // TODO(hyeontaek): Factor some of them out as `XlaCompatibleExecutable`.
  virtual int num_devices() const = 0;
  virtual int64_t SizeOfGeneratedCodeInBytes() const = 0;
  virtual StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const = 0;

  // The following APIs are taken from `xla::PjRtLoadedExecutable` for fast
  // prototyping.

  // TODO(hyeontaek): Move the following to pjrt_executable.h and put it in an
  // `XlaCompatibleExecutable`.
  // Returns a list of parameter Sharding.
  virtual std::optional<std::vector<OpSharding>> GetParameterShardings()
      const = 0;
  // Returns a list of output OpSharding.
  virtual std::optional<std::vector<OpSharding>> GetOutputShardings() const = 0;
  // Return an HloModule (optimized) per partition.
  virtual StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const = 0;

  // `LoadedExecutable` methods.

  // Short-term alias.
  using ExecuteOptions = ::xla::ExecuteOptions;

  // Result from an execution.
  struct ExecuteResult {
    // Resulting status of the execution.
    Future<Status> status;
    // Output arrays.
    std::vector<std::unique_ptr<Array>> outputs;
  };

  // Executes the executable on devices.
  //
  // The runtime expects input arrays to be present on the execution devices.
  //
  // If `devices` is specified, the execution runs on the devices if the runtime
  // supports. Otherwise, the execution runs on the devices where the executable
  // has been compiled and loaded onto.
  //
  // TODO(hyeontaek): This call does not have strict "barrier" semantics, and
  // thus it is up to the backend implementation: Some backends will wait all
  // arguments to be available to run any computation (which may be composed of
  // individually dispatchable sub-computations), while others may run the
  // computation incrementally. Some backends will mark outputs to become ready
  // roughly at the same time, while others may make outputs ready
  // incrementally. We need to have a stricter way to control this behavior
  // (e.g., having per-argument/output booleans or providing a separate barrier
  // API).
  virtual StatusOr<ExecuteResult> Execute(
      absl::Span<Array* const> args, const ExecuteOptions& options,
      std::optional<DeviceList> devices) = 0;

  // Deletes the executable from the devices. The operation may be asynchronous.
  // The returned future will have the result of the deletion on the devices.
  // Implementations that do not track the completion of the deletion operation
  // may make the future immediately ready with an OK status.
  virtual Future<Status> Delete() = 0;
  // Returns whether the executable has been enqueued for deletion from the
  // devices.
  virtual bool IsDeleted() const = 0;

  // The following APIs are taken from xla::PjRtLoadedExecutable for fast
  // prototyping.
  // TODO(hyeontaek): Move the following XLA-specific methods to
  // pjrt_executable.h and put it in an `XlaCompatibleExecutable`.

  virtual const DeviceAssignment& device_assignment() const = 0;
  using LogicalDeviceIds = ::xla::PjRtLoadedExecutable::LogicalDeviceIds;
  virtual absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const = 0;
  virtual absl::Span<Device* const> addressable_devices() const = 0;

  static char ID;  // NOLINT
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_EXECUTABLE_H_
