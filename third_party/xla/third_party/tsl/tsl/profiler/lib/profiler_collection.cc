/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_collection.h"

#include <any>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

ProfilerCollection::ProfilerCollection(
    std::vector<std::unique_ptr<ProfilerInterface>> profilers)
    : profilers_(std::move(profilers)) {}

absl::Status ProfilerCollection::Start() {
  absl::Status status;
  for (auto& profiler : profilers_) {
    status.Update(profiler->Start());
  }
  return status;
}

absl::Status ProfilerCollection::Stop() {
  absl::Status status;
  for (auto& profiler : profilers_) {
    status.Update(profiler->Stop());
  }
  return status;
}

absl::Status ProfilerCollection::CollectData(
    tensorflow::profiler::XSpace* space) {
  absl::Status status;
  for (auto& profiler : profilers_) {
    status.Update(profiler->CollectData(space));
  }
  profilers_.clear();  // data has been collected
  return status;
}

absl::StatusOr<ConsumeResult> ProfilerCollection::Consume() {
  std::vector<std::any> data_vector;
  data_vector.reserve(profilers_.size());
  size_t total_estimated_size_bytes = 0;

  for (auto& profiler : profilers_) {
    auto result = profiler->Consume();
    if (result.ok()) {
      data_vector.push_back(std::move(result->data));
      total_estimated_size_bytes += result->estimated_size_bytes;
    } else {
      LOG(ERROR) << "Profiler consume failed: " << result.status();
      data_vector.push_back(std::any());
    }
  }

  return ConsumeResult{std::any(std::move(data_vector)),
                       total_estimated_size_bytes};
}

absl::Status ProfilerCollection::Serialize(
    std::any data, tensorflow::profiler::XSpace* space) {
  auto* data_vector_ptr = std::any_cast<std::vector<std::any>>(&data);
  if (data_vector_ptr == nullptr) {
    return absl::InvalidArgumentError(
        "Invalid data type for ProfilerCollection::Serialize");
  }

  if (data_vector_ptr->size() != profilers_.size()) {
    return absl::InternalError(
        "Data vector size mismatch in ProfilerCollection::Serialize");
  }

  absl::Status status;
  for (size_t i = 0; i < profilers_.size(); ++i) {
    if ((*data_vector_ptr)[i].has_value()) {
      status.Update(
          profilers_[i]->Serialize(std::move((*data_vector_ptr)[i]), space));
    }
  }
  return status;
}

}  // namespace profiler
}  // namespace tsl
