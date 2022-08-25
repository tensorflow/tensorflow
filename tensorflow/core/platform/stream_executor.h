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

#ifndef TENSORFLOW_PLATFORM_STREAM_EXECUTOR_H_
#define TENSORFLOW_PLATFORM_STREAM_EXECUTOR_H_

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.h"
#include "tensorflow/compiler/xla/stream_executor/event.h"
#include "tensorflow/compiler/xla/stream_executor/host/host_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/lib/status.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/platform/dso_loader.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_platform_id.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/platform/platform.h"

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(jlebar): Remove this once we've completed
// the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

#endif  // TENSORFLOW_PLATFORM_STREAM_EXECUTOR_H_
