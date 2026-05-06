/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_TSL_PROFILER_LIB_PROFILER_ORCHESTRATOR_H_
#define TENSORFLOW_TSL_PROFILER_LIB_PROFILER_ORCHESTRATOR_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

class ProfilerSessionOrchestrator {
 public:
  explicit ProfilerSessionOrchestrator(
      const tensorflow::ProfileOptions& options);
  ~ProfilerSessionOrchestrator();

  absl::Status Start();

  absl::StatusOr<int> Consume();

  absl::Status Serialize(int buffer_index);

  absl::Status Stop();

  void ClearConsumeBuffers();

  const std::vector<uint8_t>& GetConsumeBuffer(int index) const {
    return consume_buffers_[index];
  }
  const tensorflow::profiler::XSpace& GetSerializeSpace() const {
    return serialize_space_;
  }

 private:
  tensorflow::ProfileOptions options_;
  std::unique_ptr<tsl::ProfilerSession> session_;
  std::vector<std::vector<uint8_t>> consume_buffers_;
  tensorflow::profiler::XSpace serialize_space_;
};

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_LIB_PROFILER_ORCHESTRATOR_H_
