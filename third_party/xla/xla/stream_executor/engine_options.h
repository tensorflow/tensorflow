/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
------------------------------------------------------------------------------*/

#ifndef XLA_STREAM_EXECUTOR_ENGINE_OPTIONS_H_
#define XLA_STREAM_EXECUTOR_ENGINE_OPTIONS_H_

namespace stream_executor {

// Options (used when calling math libraries) that specify the behavior of
// operations like matrix multiplications and convolutions.
struct EngineOptions {
  EngineOptions(bool require_determinism, bool allow_tf32,
                bool require_command_buffer)
      : require_determinism(require_determinism),
        allow_tf32(allow_tf32),
        require_command_buffer(require_command_buffer) {}

  EngineOptions()
      : require_determinism(false),
        allow_tf32(true),
        require_command_buffer(false) {}

  // If true, the op must be deterministic
  bool require_determinism;
  // If true, float32 inputs can be rounded to TensorFloat-32 precision
  bool allow_tf32;
  // If true, the execution plan selected must support command buffer
  // construction.
  bool require_command_buffer;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ENGINE_OPTIONS_H_
