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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

class LocalExecutable {
 public:
  // Low-level constructor; LocalClient::Compile() is the usual way to create
  // executables.
  LocalExecutable(std::unique_ptr<Executable> executable, Backend* backend,
                  ExecutableBuildOptions build_options);

  // Run the compiled computation with the given arguments and options and
  // return the result.
  StatusOr<ScopedShapedBuffer> Run(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to Run(), but allows for donating argument buffers to the
  // executable.
  StatusOr<ExecutionOutput> Run(std::vector<ExecutionInput> arguments,
                                ExecutableRunOptions run_options);

  // Similar to Run(), but need not block the host waiting for the computation
  // to complete before returning.
  StatusOr<ScopedShapedBuffer> RunAsync(
      const absl::Span<const ShapedBuffer* const> arguments,
      ExecutableRunOptions run_options);

  // Similar to RunAsync(), but allows for donating argument buffers to the
  // executable.
  StatusOr<ExecutionOutput> RunAsync(std::vector<ExecutionInput> arguments,
                                     ExecutableRunOptions run_options);

  // Return the options used to build the executable.
  const ExecutableBuildOptions& build_options() const { return build_options_; }

  // Return the built executable.
  Executable* executable() const { return executable_.get(); }

 private:
  StatusOr<ExecutionOutput> RunAsync(
      absl::Span<Shape const* const> argument_host_shapes,
      std::vector<ExecutionInput> arguments, ExecutableRunOptions run_options);

  // Validates that the given arguments and options satisfy various constraints
  // of the computation.
  //
  // The given ExecutableRunOptions override any values from TF_XLA_FLAGS
  // environment variable.
  Status ValidateExecutionOptions(const ExecutableRunOptions& run_options,
                                  const Backend& backend);

  // Returns a literal containing the contents of the given ShapedBuffer.
  StatusOr<Literal> LiteralFromShapedBuffer(const ShapedBuffer& shaped_buffer);

  StatusOr<std::pair<ServiceExecutableRunOptions, StreamPool::Ptr>> RunHelper(
      const absl::Span<const Shape* const> argument_shapes,
      ExecutableRunOptions run_options);

  // The ordinal of the device which this executable was compiled for. The
  // executable can run on all equivalent devices (as determined by
  // Backend::devices_equivalent).
  int build_device_ordinal() const { return build_options_.device_ordinal(); }

  template <typename T>
  StatusOr<T> AsyncCallAndBlockHostUntilDone(
      absl::Span<Shape const* const> argument_shapes,
      const ExecutableRunOptions& run_options,
      std::function<StatusOr<T>(const ExecutableRunOptions&)> async_callback) {
    TF_ASSIGN_OR_RETURN(auto options_and_stream,
                        RunHelper(argument_shapes, run_options));
    ExecutableRunOptions options = options_and_stream.first.run_options();
    options.set_device_ordinal(-1);
    StatusOr<T> result = async_callback(options);
    Status block_status = options.stream()->BlockHostUntilDone();
    TF_RETURN_IF_ERROR(result.status());
    TF_RETURN_IF_ERROR(block_status);
    return result;
  }

  // Compiled computation.
  std::unique_ptr<Executable> executable_;

  // Execution backend.
  Backend* backend_ = nullptr;

  // Options used to build the executable.
  const ExecutableBuildOptions build_options_;
};

// An XLA Client specialization for use when the client and service run in
// the same process.
class LocalClient : public Client {
 public:
  explicit LocalClient(LocalService* service)
      : Client(service), local_service_(service) {}

  LocalClient(const LocalClient&) = delete;
  void operator=(const LocalClient&) = delete;

  // Build and return LocalExecutable objects (one per partition, as specified
  // by the build options). The executable is compiled using the given
  // XlaComputation, argument layouts and options.
  //
  // The given ExecutableBuildOptions overrides any values from XLA_FLAGS
  // environment variable.
  StatusOr<std::vector<std::unique_ptr<LocalExecutable>>> Compile(
      const XlaComputation& computation,
      const absl::Span<const Shape* const> argument_layouts,
      const ExecutableBuildOptions& options);

  // Same as Compile() above, but return AotCompilationResult objects (instead
  // of LocalExecutable objects), which can be persisted to later load
  // LocalExecutable(s) using the Load() method below.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(const XlaComputation& computation,
                     const absl::Span<const Shape* const> argument_layouts,
                     const ExecutableBuildOptions& options);

  // Return a LocalExecutable object loaded from a serialized
  // AotCompilationResult.
  StatusOr<std::unique_ptr<LocalExecutable>> Load(
      const std::string& serialized_aot_result,
      const ExecutableBuildOptions& options);

  // Copy the literal data to the device with the given ordinal and return as a
  // ScopedShapedBuffer. If non-null the given memory allocator is used for
  // device memory allocation. If null, the default memory allocator for the
  // device is used.
  StatusOr<ScopedShapedBuffer> LiteralToShapedBuffer(
      const LiteralSlice& literal, int device_ordinal,
      se::DeviceMemoryAllocator* allocator = nullptr);

  // Transfer the BorrowingLiteral to the device with the given ordinal.
  StatusOr<TransferToServerResponse> TransferToLocalServer(
      const ::xla::BorrowingLiteral& literal, int device_ordinal);

  // Copy the data from the device contained in the given ShapedBuffer and
  // return as a Literal.
  StatusOr<Literal> ShapedBufferToLiteral(const ShapedBuffer& shaped_buffer);

  // Converts a GlobalDataHandle into a pointer to a ShapedBuffer that's valid
  // as long as the handle is valid.
  StatusOr<const ShapedBuffer*> GlobalDataToShapedBuffer(
      const GlobalDataHandle& data, int replica_number);

  // Transfer the given literal to the infeed queue of the given device.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferToInfeed.
  Status TransferToInfeedLocal(const LiteralSlice& literal, int device_ordinal);

  // Transfer and return a value from the outfeed of the given device. The
  // shape of the object to transfer is determined by `literal`'s shape.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferFromOutfeed.
  Status TransferFromOutfeedLocal(int device_ordinal,
                                  MutableBorrowingLiteral literal);

  // Returns the device ordinal that corresponds to the given replica number.
  //
  // This returns an error if there is not a one-to-one correspondence of
  // replicas to device ordinals, but is useful as a short term mechanism for
  // the "easy" case where a single replica is a single device.
  StatusOr<int> ReplicaNumberToDeviceOrdinal(int replica_number);

  // Returns the platform that the underlying service targets.
  se::Platform* platform() const;

  // Returns the number of devices on the system of the service platform
  // type. Not all devices may be supported by the service (see
  // device_ordinal_supported method).
  int device_count() const;

  // Returns the default device ordinal that the service will run computations
  // on if no device ordinal is specified in execute options.
  int default_device_ordinal() const;

  // Returns whether the device with the given ordinal can be used by the
  // service to execute computations. Not all devices of a particular platform
  // may be usable by the service (eg, a GPU with insufficient CUDA compute
  // capability).
  bool device_ordinal_supported(int device_ordinal) const;

  // Returns the backend used to execute computations.
  const Backend& backend() const;
  Backend* mutable_backend();

 private:
  LocalService* local_service_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LOCAL_CLIENT_H_
