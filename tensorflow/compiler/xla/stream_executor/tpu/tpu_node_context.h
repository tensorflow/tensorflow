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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_

#include <string>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/status_helper.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/tsl/platform/macros.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tpu {

// A TpuNodeContext object represents a specific TPU node (core). The static
// class methods represent host-wide actions.
//
// First call Initialize in a freshly reset system. Then call Create to talk to
// individual nodes.
class TpuNodeContext final {
 public:
  template <typename T>
  using StatusOr = tsl::StatusOr<T>;

  static StatusOr<std::unique_ptr<TpuNodeContext>> Create(int device_ordinal);

  explicit TpuNodeContext(int device_ordinal, XLA_TpuNodeContext* node_context)
      : device_ordinal_(device_ordinal), node_context_(node_context) {
    CHECK_NE(node_context, nullptr);
  }
  ~TpuNodeContext();

  static tsl::Status StopChipHeartbeats();

  static tsl::Status CloseTpuHost();

  static tsl::Status Initialize(int device_ordinal);

  static TpuPlatformInterface* platform();

  int device_ordinal() const;

  xla::Backend* backend() const;

  stream_executor::StreamExecutor* stream_executor() const;

  bool CompactionSupported(int device_ordinal) const;

 private:
  const int device_ordinal_;
  XLA_TpuNodeContext* const node_context_;

  TF_DISALLOW_COPY_AND_ASSIGN(TpuNodeContext);
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_TPU_TPU_NODE_CONTEXT_H_
