/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_THUNK_PASS_PIPELINE_H_
#define XLA_SERVICE_GPU_TRANSFORMS_THUNK_PASS_PIPELINE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class ThunkPassInterface {
 public:
  virtual ~ThunkPassInterface() = default;

  virtual absl::StatusOr<bool> Run(
      SequentialThunk* root_thunk, const DebugOptions& debug_options,
      const se::DeviceDescription& device_info) = 0;

  virtual absl::string_view name() const = 0;
};

class ThunkPassPipeline : public ThunkPassInterface {
 public:
  explicit ThunkPassPipeline(absl::string_view name) : name_(name) {}

  void AddPass(std::unique_ptr<ThunkPassInterface> pass) {
    passes_.push_back(std::move(pass));
  }

  absl::string_view name() const override { return name_; }

  // Runs all optimization passes on the given thunk sequence.
  // Returns true if any pass changed the thunk tree.
  absl::StatusOr<bool> Run(SequentialThunk* root_thunk,
                           const DebugOptions& debug_options,
                           const se::DeviceDescription& device_info) override;

 private:
  std::string name_;
  std::vector<std::unique_ptr<ThunkPassInterface>> passes_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_THUNK_PASS_PIPELINE_H_
