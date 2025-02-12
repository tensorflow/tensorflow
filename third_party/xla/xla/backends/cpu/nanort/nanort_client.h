/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_NANORT_NANORT_CLIENT_H_
#define XLA_BACKENDS_CPU_NANORT_NANORT_CLIENT_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/nanort/nanort_executable.h"
#include "xla/hlo/builder/xla_computation.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

// A client for compiling XLA programs to executables using the XLA:CPU backend.
class NanoRtClient {
 public:
  NanoRtClient();

  // Compiles the given XLA computation to a NanoRtExecutable using the XLA:CPU
  // backend.
  absl::StatusOr<std::unique_ptr<NanoRtExecutable>> Compile(
      const XlaComputation& computation);

 private:
  // Thread pool for running XLA:CPU compute tasks.
  std::shared_ptr<tsl::thread::ThreadPool> intra_op_thread_pool_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_NANORT_NANORT_CLIENT_H_
