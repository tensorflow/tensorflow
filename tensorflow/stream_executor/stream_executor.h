/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// The StreamExecutor is a single-device abstraction for:
//
// * Loading/launching data-parallel-kernels
// * Invoking pre-canned high-performance library routines (like matrix
//   multiply)
//
// The appropriately-typed kernel and "loader spec" are automatically generated
// for the user within a namespace by the gcudacc compiler output, so typical
// use looks like so:
//
//    namespace gpu = ::perftools::gputools;
//    namespace gcudacc = ::platforms::gpus::gcudacc;
//
//    gpu::StreamExecutor stream_exec{PlatformKind::kCuda};
//    gcudacc::kernel::MyKernel my_kernel{&stream_exec};
//    bool ok = stream_exec.GetKernel(gcudacc::spec::MyKernelSpec(),
//    &my_kernel);
//    if (!ok) { ... }
//    gpu::DeviceMemory<int> result = stream_exec.AllocateZeroed<int>();
//    if (result == nullptr) { ... }
//    int host_result;
//    gpu::Stream my_stream{&stream_exec};
//    my_stream
//      .Init()
//      .ThenLaunch(ThreadDim{1024}, BlockDim{1}, my_kernel, result)
//      .ThenMemcpy(&host_result, result, sizeof(host_result))
//      .BlockHostUntilDone()
//    if (!my_stream.ok()) { ... }
//    printf("%d\n", host_result);
//
// Since the device may operate asynchronously to the host, the
// Stream::BlockHostUntilDone() call forces the calling host thread to wait for
// the chain of commands specified for the Stream to complete execution.

#ifndef TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
#define TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_H_

#include "tensorflow/stream_executor/device_description.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/device_memory.h"    // IWYU pragma: export
#include "tensorflow/stream_executor/device_options.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/event.h"           // IWYU pragma: export
#include "tensorflow/stream_executor/kernel.h"       // IWYU pragma: export
#include "tensorflow/stream_executor/kernel_spec.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/launch_dim.h"   // IWYU pragma: export
#include "tensorflow/stream_executor/multi_platform_manager.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/platform.h"     // IWYU pragma: export
#include "tensorflow/stream_executor/stream.h"       // IWYU pragma: export
#include "tensorflow/stream_executor/stream_executor_pimpl.h"  // IWYU pragma: export
#include "tensorflow/stream_executor/timer.h"  // IWYU pragma: export

#endif  // TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_H_
