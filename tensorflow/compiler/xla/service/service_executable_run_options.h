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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/service/pool.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace xla {

// Class containing options for running a LocalExecutable and other auxiliary
// data, now only a stream cache for GPU backend.
class ServiceExecutableRunOptions {
 public:
  using StreamBorrower =
      std::function<StatusOr<Pool<perftools::gputools::Stream>::SmartPtr>(int)>;

  ServiceExecutableRunOptions()
      : ServiceExecutableRunOptions(ExecutableRunOptions()) {}

  explicit ServiceExecutableRunOptions(
      ExecutableRunOptions run_options, StreamBorrower borrow_stream = nullptr,
      tensorflow::thread::ThreadPool* xla_intra_op_thread_pool = nullptr)
      : run_options_(std::move(run_options)),
        borrow_stream_(std::move(borrow_stream)),
        xla_intra_op_thread_pool_(xla_intra_op_thread_pool) {}

  // Returns reference or pointer to `ExecutableRunOptions` member.
  const ExecutableRunOptions& run_options() const { return run_options_; }
  ExecutableRunOptions* mutable_run_options() { return &run_options_; }

  // Delegate to `ExecutableRunOptions` member.
  perftools::gputools::Stream* stream() const { return run_options_.stream(); }
  DeviceMemoryAllocator* allocator() const { return run_options_.allocator(); }
  int device_ordinal() const { return run_options_.device_ordinal(); }

  // Borrows a stream and returns a smart pointer which returns the stream on
  // destruction.
  StatusOr<Pool<perftools::gputools::Stream>::SmartPtr> BorrowStream(
      int device_ordinal) const {
    return borrow_stream_
               ? borrow_stream_(device_ordinal)
               : Status(tensorflow::error::UNIMPLEMENTED, "No stream cache");
  }

  // Returns reference to thread pool for execution of XLA ops on CPU backend.
  tensorflow::thread::ThreadPool* xla_intra_op_thread_pool() const {
    return xla_intra_op_thread_pool_;
  }

 private:
  ExecutableRunOptions run_options_;
  StreamBorrower borrow_stream_;
  tensorflow::thread::ThreadPool* xla_intra_op_thread_pool_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_
