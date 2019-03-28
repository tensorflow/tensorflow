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

#ifndef TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_

// These classes are forward declared so that ExecutableRunOptions can be linked
// into an XLA-compiled binary without having to link all of the pointed-to
// objects (e.g., for an ahead-of-time compiled CPU binary, the gpu tools don't
// need to be linked).
namespace stream_executor {
class Stream;
class Platform;
}  // namespace stream_executor

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla {

class DeviceMemoryAllocator;
class DeviceAssignment;
class ExecutionProfile;

// Class containing options for running a LocalExecutable.
class ExecutableRunOptions {
 public:
  // Specifies the allocator to use during execution.
  ExecutableRunOptions& set_allocator(DeviceMemoryAllocator* allocator);
  DeviceMemoryAllocator* allocator() const;

  // If set, this is the device to run the computation on. Valid device_ordinal
  // values are: 0 to # of devices - 1. These values are identical to the device
  // ordinal values used by StreamExecutor. The device must be of the same type
  // as the executable was compiled for. A value of -1 indicates this option has
  // not been set.
  ExecutableRunOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this is the stream to run the computation on. The platform of the
  // stream must match the platform the executable was built for.  A value of
  // nullptr indicates the option has not been set.
  ExecutableRunOptions& set_stream(stream_executor::Stream* stream);
  stream_executor::Stream* stream() const;

  // If set, this is the stream to perform any pre-computation transfers on.
  // The platform of the stream must match the platform the executable was
  // built for.  A value of nullptr indicates the option has not been set.
  ExecutableRunOptions& set_host_to_device_stream(
      stream_executor::Stream* stream);
  stream_executor::Stream* host_to_device_stream() const;

  // Sets the thread pool device on which to run Eigen subcomputations.
  //
  // This field must be set for XLA:CPU models that call Eigen routines, but may
  // be null otherwise.  Routines that use this field should always CHECK (or
  // TF_RET_CHECK) that it's not null before dereferencing it, so that users get
  // a clean crash rather than a segfault.
  //
  // Does not take ownership.
  ExecutableRunOptions& set_intra_op_thread_pool(
      const Eigen::ThreadPoolDevice* intra_op_thread_pool);
  const Eigen::ThreadPoolDevice* intra_op_thread_pool() const;

  // If set, profiling information is written to 'profile'.
  ExecutionProfile* execution_profile() const;
  ExecutableRunOptions& set_execution_profile(ExecutionProfile* profile);

  ExecutableRunOptions& set_device_assignment(
      const DeviceAssignment* device_assignment);
  const DeviceAssignment* device_assignment() const;

  ExecutableRunOptions& set_rng_seed(int rng_seed);
  int rng_seed() const;

 private:
  DeviceMemoryAllocator* allocator_ = nullptr;
  int device_ordinal_ = -1;
  const DeviceAssignment* device_assignment_ = nullptr;
  stream_executor::Stream* stream_ = nullptr;
  const Eigen::ThreadPoolDevice* intra_op_thread_pool_ = nullptr;
  ExecutionProfile* execution_profile_ = nullptr;
  int rng_seed_ = 0;
  stream_executor::Stream* host_to_device_stream_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_
