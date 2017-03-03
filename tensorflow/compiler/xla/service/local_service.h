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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOCAL_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOCAL_SERVICE_H_

#include <memory>

#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/service.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Computation execution options which may be set by the client when executing
// locally (via LocalClient::ExecuteLocally).
class LocalExecuteOptions {
 public:
  // Specifies the allocator to use during execution. Execution will fail if no
  // allocator is provided.
  LocalExecuteOptions& set_allocator(DeviceMemoryAllocator* allocator);
  DeviceMemoryAllocator* allocator() const;

  // If set, this is the platform to run the computation on. This must match
  // the underlying platform of the service. A value of nullptr means the
  // platform is not set.
  // TODO(b/28616830): Support multiple platforms.
  LocalExecuteOptions& set_platform(perftools::gputools::Platform* platform);
  perftools::gputools::Platform* platform() const;

  // If set, this is the device to run the computation on. Valid device_ordinal
  // values are: 0 to # of devices - 1. These values are identical to the
  // device ordinal values used by StreamExecutor. A value of < 0 means the
  // ordinal is not set.
  LocalExecuteOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this is the stream to run the computation on. The platform of the
  // stream must match the service's platform. The device ordinal
  // option (if set) must match the stream's device. A value of nullptr means
  // the stream is not set.
  LocalExecuteOptions& set_stream(perftools::gputools::Stream* stream);
  perftools::gputools::Stream* stream() const;

  // If set, collect profile information during execution and fill the given
  // ExecutionProfile object with the profile data. A value of nullptr means
  // the profile is not set.
  LocalExecuteOptions& set_execution_profile(ExecutionProfile* profile);
  ExecutionProfile* execution_profile() const;

  // If set, this specifies the layout of the result of the computation. If not
  // set, the service will chose the layout of the result. A Shape is used to
  // store the layout to accomodate tuple result shapes. A value of nullptr
  // means the shape is not set.
  LocalExecuteOptions& set_result_layout(const Shape& shape_with_layout);
  const Shape* result_layout() const;

 private:
  DeviceMemoryAllocator* allocator_ = nullptr;
  perftools::gputools::Platform* platform_ = nullptr;
  int device_ordinal_ = -1;
  perftools::gputools::Stream* stream_ = nullptr;
  ExecutionProfile* profile_ = nullptr;

  bool has_result_shape_with_layout_ = false;
  Shape result_shape_with_layout_;
};

// Service implementation that extends the XLA Service to leverage running
// in the same process as the client.
class LocalService : public Service {
 public:
  // Factory for creating a LocalService. The parameter platform is the platform
  // that the service should target. If platform is null then the default
  // platform is used.
  static StatusOr<std::unique_ptr<LocalService>> NewService(
      perftools::gputools::Platform* platform);
  static StatusOr<std::unique_ptr<LocalService>> NewService(
      const ServiceOptions& options);

  // For an array of arguments, validate that each is placed on the
  // specified device_ordinal, and return the DeviceMemoryBase
  // corresponding to each argument.
  tensorflow::Status ResolveArguments(
      const tensorflow::gtl::ArraySlice<const GlobalDataHandle*> arguments,
      int device_ordinal,
      std::vector<perftools::gputools::DeviceMemoryBase>* argument_ptrs);

  // Return a handle to a buffer large enough to hold shape, allocated
  // on device_ordinal. If allocate_space_for_deep_copy, the buffer is
  // large enough to hold all sub-buffers of a tuple shape, otherwise
  // it is only as large as the top-level tuple pointer array.
  StatusOr<GlobalDataHandle> AllocateBufferOnDevice(
      const Shape& shape, int device_ordinal,
      bool allocate_space_for_deep_copy);

  // Execute the given computation with the given arguments and options with
  // zero-copy data handling of arguments and result.
  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteLocally(
      const ComputationHandle& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const LocalExecuteOptions& options);

  // Overload which writes the result into the given ShapedBuffer "result".
  // Due to aliasing, not all buffers which comprise "result" may be utilized
  // in the computation and thus be uninitialized.  The |ShapedBuffer::buffer|
  // or |ShapedBuffer::mutable_buffer| methods should be used to map an index to
  // the initialized buffer.
  //
  // For example:
  //  Let 'result' be a ShapedBuffer holding a tuple with the same element,
  //  'x', twice: (x, x).  It is incorrect to assume that the second buffer
  //  which comprises 'result' is initialized.  Instead, a mapping has been
  //  added to 'result' which can be used to recover the correct buffer.
  //  In this case, result->buffer({0}) should be used to extract the address of
  //  the first tuple element while result->buffer({1}) should be used for the
  //  second.
  tensorflow::Status ExecuteLocally(
      const ComputationHandle& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const LocalExecuteOptions& options, ShapedBuffer* result_buffer);

  // A description of a computation to compile using CompileAheadOfTime.
  struct AheadOfTimeComputationInstance {
    ComputationHandle computation;
    std::vector<const Shape*> argument_layouts;
    const Shape* result_layout = nullptr;
  };

  // Compiles a list of computations for ahead-of-time execution.  This is
  // intended for use in static compilation.  See
  // |LocalClient::CompileAheadOfTime| for additional details.
  StatusOr<std::vector<std::unique_ptr<AotCompilationResult>>>
  CompileAheadOfTime(
      const tensorflow::gtl::ArraySlice<AheadOfTimeComputationInstance>
          computations,
      const AotCompilationOptions& Options);

  // Builds an Executable with the given argument layouts and options. If
  // result_layout is non-null, then the executable is compiled to produce a
  // result of the given layout.
  StatusOr<std::unique_ptr<Executable>> CompileExecutable(
      const ComputationHandle& computation,
      const tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
      const Shape* result_layout, int device_ordinal, bool has_hybrid_result);

 private:
  explicit LocalService(std::unique_ptr<Backend> backend,
                        std::unique_ptr<Backend> compute_constant_backend);
  LocalService(const LocalService&) = delete;
  void operator=(const LocalService&) = delete;

  // Internal helper for executing a computation. If result_buffer is null then
  // the result is returned as a ShapedBuffer. If result_buffer is non-null then
  // the result is written into result_buffer and a null ShapedBuffer pointer is
  // returned.
  StatusOr<std::unique_ptr<ShapedBuffer>> ExecuteLocallyInternal(
      const ComputationHandle& computation,
      tensorflow::gtl::ArraySlice<const ShapedBuffer*> arguments,
      const LocalExecuteOptions& options,
      ShapedBuffer* preallocated_result_buffer);

  // Validates the given options and argument layouts and returns an appropriate
  // error code.
  tensorflow::Status ValidateExecuteOptions(
      const ProgramShape& program_shape,
      tensorflow::gtl::ArraySlice<const Shape*> arguments,
      const LocalExecuteOptions& options,
      const ShapedBuffer* preallocated_result_buffer);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOCAL_SERVICE_H_
