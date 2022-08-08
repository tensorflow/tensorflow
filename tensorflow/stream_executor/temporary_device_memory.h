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

// Temporary memories are used to allocate scratch space required by an
// operation about to be enqueued onto a stream.
//
//    std::unique_ptr<TemporaryDeviceMemory<float>> temporary_memory =
//        stream.AllocateTemporaryArray<float>(1024).ConsumeValueOrDie();
//    // ... enqueue stuff onto the stream using the temporary memory ...
//    // Note that the memory is accessible via
//    // temporary_memory->device_memory() and similar.
//
//    // Finalize the temporary memory. The underlying device memory may
//    // be released any time after this program point, as another thread may
//    // call Stream::BlockHostUntilDone, causing synchronization. This
//    // finalization also happens automatically for the user if the unique_ptr
//    // goes out of scope.
//    temporary_memory.Finalize();
//
// WARNING: do NOT hold onto the device memory associated with temporary_memory
// after finalization. If temporary_memory->device_memory() is used after the
// temporary memory is finalized, it will cause a DCHECK failure.
//
// Note that standard usage takes advantage of the type-safe wrapper,
// TemporaryDeviceMemory<T>, defined below.
//
// Also see tests for executable sample usage.

#ifndef TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
#define TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_

#include "tensorflow/compiler/xla/stream_executor/temporary_device_memory.h"

#endif  // TENSORFLOW_STREAM_EXECUTOR_TEMPORARY_DEVICE_MEMORY_H_
