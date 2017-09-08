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

// Service implementation that extends the XLA Service to leverage running
// in the same process as the client.
class LocalService : public Service {
 public:
  // Factory for creating a LocalService.
  static StatusOr<std::unique_ptr<LocalService>> NewService(
      const ServiceOptions& options);

  // Return a handle to a buffer large enough to hold shape, allocated
  // on device_ordinal. If allocate_space_for_deep_copy, the buffer is
  // large enough to hold all sub-buffers of a tuple shape, otherwise
  // it is only as large as the top-level tuple pointer array.
  StatusOr<GlobalDataHandle> AllocateBufferOnDevice(
      const Shape& shape, int device_ordinal,
      bool allocate_space_for_deep_copy);

  // Builds an Executable with the given argument layouts and options. If
  // result_layout is non-null, then the executable is compiled to produce a
  // result of the given layout.
  StatusOr<std::unique_ptr<Executable>> CompileExecutable(
      const ComputationHandle& computation,
      const tensorflow::gtl::ArraySlice<const Shape*> argument_layouts,
      const Shape* result_layout, int device_ordinal, bool has_hybrid_result);

 private:
  explicit LocalService(const ServiceOptions& options,
                        std::unique_ptr<Backend> backend);
  LocalService(const LocalService&) = delete;
  void operator=(const LocalService&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOCAL_SERVICE_H_
