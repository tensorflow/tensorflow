/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_C_API_DEFN_H_
#define XLA_STREAM_EXECUTOR_TPU_C_API_DEFN_H_

#include "xla/stream_executor/device_options.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

// Definitions for XLA API data structures. Any underlying C++ data structures
// are implementation details and should only be used from within the stream
// executor implementation.

namespace stream_executor {
class Platform;
class StreamExecutor;
}  // namespace stream_executor

struct SE_Platform {
  stream_executor::Platform* platform;
};

struct SE_StreamExecutor {
  stream_executor::StreamExecutor* executor;
};

struct SE_Stream {
  explicit SE_Stream(stream_executor::StreamExecutor* parent)
      : stream(parent) {}
  stream_executor::Stream stream;
};

struct SE_Event {
  explicit SE_Event(stream_executor::StreamExecutor* parent) : event(parent) {}
  stream_executor::Event event;
};

struct SE_StreamExecutorConfig {
  stream_executor::StreamExecutorConfig config;
};

struct SE_DeviceOptions {
  stream_executor::DeviceOptions options;
};

// Ignored -- these are just used to enforce the interface types
struct XLA_TransferManager {};
struct XLA_ComputationPlacer {};
struct SE_TpuTopology {};
struct SE_TpuTopology_Core {};

#endif  // XLA_STREAM_EXECUTOR_TPU_C_API_DEFN_H_
