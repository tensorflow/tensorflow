/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/command_queue.h"

#include "absl/memory/memory.h"
#include "tensorflow/lite/delegates/gpu/common/gpu_info.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_call.h"
#include "tensorflow/lite/delegates/gpu/gl/gl_sync.h"
#include "tensorflow/lite/delegates/gpu/gl/portable_gl31.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class DefaultCommandQueue : public CommandQueue {
 public:
  absl::Status Dispatch(const GlProgram& program,
                        const uint3& workgroups) override {
    RETURN_IF_ERROR(program.Dispatch(workgroups));
    return TFLITE_GPU_CALL_GL(glMemoryBarrier, GL_ALL_BARRIER_BITS);
  }

  absl::Status WaitForCompletion() override {
    // TODO(akulik): Maybe let the user choose which wait method to use.
    return GlActiveSyncWait();
  }

  absl::Status Flush() override { return absl::OkStatus(); }
};

// On Adreno do flush periodically as this affects performance. Command queue
// needs to be manually managed to ensure that accumulated work goes to GPU as
// fast as it can.
//
// Also, on older Adreno devices glFlush is required after every memory barrier
// to avoid hitting GPU driver bug.
class AdrenoCommandQueue : public DefaultCommandQueue {
 public:
  explicit AdrenoCommandQueue(int flush_every_n)
      : flush_every_n_(flush_every_n) {}

  absl::Status Dispatch(const GlProgram& program,
                        const uint3& workgroups) final {
    RETURN_IF_ERROR(DefaultCommandQueue::Dispatch(program, workgroups));
    if ((++program_counter_ % flush_every_n_) == 0) {
      glFlush();
    }
    return absl::OkStatus();
  }

  absl::Status WaitForCompletion() override {
    program_counter_ = 0;
    return DefaultCommandQueue::WaitForCompletion();
  }

  absl::Status Flush() final {
    // Flush exactly once after the last dispatch.
    if (program_counter_ != 0) {
      program_counter_ = 0;
      glFlush();
    }
    return absl::OkStatus();
  }

 private:
  const int flush_every_n_;
  int program_counter_ = 0;
};

}  // namespace

std::unique_ptr<CommandQueue> NewCommandQueue(const GpuInfo& gpu_info) {
  if (gpu_info.type == GpuType::ADRENO) {
    int flush_every_n = 1;
    // On Adreno 630 and Adreno 505 there is up to 2x performance boost when
    // glFlush happens not so often.
    if (gpu_info.gpu_model == GpuModel::ADRENO630 ||
        gpu_info.gpu_model == GpuModel::ADRENO505) {
      flush_every_n = 10;
    }
    return absl::make_unique<AdrenoCommandQueue>(flush_every_n);
  }
  return absl::make_unique<DefaultCommandQueue>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
