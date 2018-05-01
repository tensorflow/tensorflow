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

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/computation.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/local_service.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

class LocalExecutable {
 public:
  // Run the compiled computation with the given arguments and options and
  // return the result.
  StatusOr<ScopedShapedBuffer> Run(
      const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      ExecutableRunOptions run_options);

  // Return the options used to build the executable.
  const ExecutableBuildOptions& build_options() const { return build_options_; }

  // Return the built executable.
  Executable* executable() const { return executable_.get(); }

 private:
  // Only a local client can construct these objects.
  friend class LocalClient;

  // Constructor invoked by LocalClient.
  LocalExecutable(std::unique_ptr<Executable> executable, Backend* backend,
                  ExecutableBuildOptions build_options);

  // Validates that the given arguments and options satisfy various constraints
  // of the computation.
  tensorflow::Status ValidateExecutionOptions(
      const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const ExecutableRunOptions& run_options, const Backend& backend);

  // Records the computation in a SessionModule proto with the arguments used to
  // invoke it, and the result. Enabled by flag: --tla_dump_executions_to.
  StatusOr<ScopedShapedBuffer> ExecuteAndDump(
      const ServiceExecutableRunOptions* run_options,
      const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments);

  // Records the arguments used to invoke the computation in a SessionModule
  // proto.
  tensorflow::Status RecordArguments(
      const tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      SessionModule* session_module);

  // Records the result of the computation in a SessionModule proto.
  tensorflow::Status RecordResult(const ShapedBuffer* result,
                                  SessionModule* session_module);

  // Returns a literal containing the contents of the given ShapedBuffer.
  StatusOr<std::unique_ptr<Literal>> LiteralFromShapedBuffer(
      const ShapedBuffer& shaped_buffer);

  // The ordinal of the device which this executable was compiled for. The
  // executable can run on all equivalent devices (as determined by
  // Backend::devices_equivalent).
  int build_device_ordinal() const { return build_options_.device_ordinal(); }

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

  // Build and return a LocalExecutable object. The executable is compiled using
  // the given argument layouts and options.
  StatusOr<std::unique_ptr<LocalExecutable>> Compile(
      const Computation& computation,
      const tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
      const ExecutableBuildOptions& options);

  // Build and return a LocalExecutable object. The executable is compiled using
  // the given XlaComputation, argument layouts and options.
  //
  // TODO(b/74197823): This is a part of a NOT YET ready refactor.
  StatusOr<std::unique_ptr<LocalExecutable>> Compile(
      const XlaComputation& computation,
      const tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
      const ExecutableBuildOptions& options);

  // Copy the literal data to the device with the given ordinal and return as a
  // ScopedShapedBuffer. If non-null the given memory allocator is used for
  // device memory allocation. If null, the default memory allocator for the
  // device is used.
  StatusOr<ScopedShapedBuffer> LiteralToShapedBuffer(
      const Literal& literal, int device_ordinal,
      DeviceMemoryAllocator* allocator = nullptr);

  // Copy the data from the device contained in the given ShapedBuffer and
  // return as a Literal.
  StatusOr<std::unique_ptr<Literal>> ShapedBufferToLiteral(
      const ShapedBuffer& shaped_buffer);

  // Transfer the given literal to the infeed queue of the given device.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferToInfeed.
  Status TransferToInfeedLocal(const Literal& literal, int device_ordinal);

  // Transfer and return a value of the given shape from the outfeed of the
  // given device.
  // TODO(b/69670845): Remove the 'Local' from the name when LocalClient does
  // not inherit from Client and there is no possibility of confusion with
  // Client::TransferFromOutfeed.
  StatusOr<std::unique_ptr<Literal>> TransferFromOutfeedLocal(
      const Shape& shape, int device_ordinal);

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
