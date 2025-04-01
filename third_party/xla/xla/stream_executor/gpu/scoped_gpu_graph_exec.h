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

#ifndef XLA_STREAM_EXECUTOR_GPU_SCOPED_GPU_GRAPH_EXEC_H_
#define XLA_STREAM_EXECUTOR_GPU_SCOPED_GPU_GRAPH_EXEC_H_

#include "xla/stream_executor/gpu/scoped_update_mode.h"

namespace stream_executor::gpu {

// ScopedGraphExec reads from `*exec_graph_location` and
// `*is_owned_graph_exec_location` in the constructor, stores their values for
// the lifetime of the object, and restores them in the destructor.
// Basically it's a simple RAII variable state wrapper for two variables with
// one being a bool and the other being a templated `GraphExecHandle`.
template <typename GraphExecHandle>
class ScopedGraphExec : public ScopedUpdateMode {
 public:
  ScopedGraphExec(GraphExecHandle* exec_graph_location,
                  bool* is_owned_graph_exec_location)
      : restore_location_(exec_graph_location),
        restore_is_owned_location_(is_owned_graph_exec_location),
        restore_value_(*exec_graph_location),
        restore_is_owned_value_(*is_owned_graph_exec_location) {}

  ~ScopedGraphExec() override {
    *restore_location_ = restore_value_;
    *restore_is_owned_location_ = restore_is_owned_value_;
  }

 private:
  GraphExecHandle* restore_location_;
  bool* restore_is_owned_location_;
  GraphExecHandle restore_value_;
  bool restore_is_owned_value_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_SCOPED_GPU_GRAPH_EXEC_H_
