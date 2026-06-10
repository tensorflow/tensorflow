/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_XLA_TRANSFORM_H_
#define XLA_SERVICE_XLA_TRANSFORM_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/hlo.pb.h"

namespace xla {

// An XlaTransform is a user-defined transformation, which is applied at a
// user-specified stage of the XLA compilation pipeline.

class XlaTransformBase {
 public:
  explicit XlaTransformBase(std::string name) : name_(std::move(name)) {}
  virtual ~XlaTransformBase() = default;

  const std::string& name() const { return name_; }

 private:
  std::string name_;
};

// HloXlaTransform is an XlaTransform that operates on an HLO module. Currently,
// it can only be applied at two stages of the XLA compilation pipeline:
// pre-scheduler and post-scheduler. We expect to expand the stages in the
// future.
class HloXlaTransform : public XlaTransformBase {
 public:
  enum class PipelineStage {
    kPreScheduler = 0,
    kPostScheduler,
  };

  explicit HloXlaTransform(std::string name)
      : XlaTransformBase(std::move(name)) {}

  // Applies the transform to the given HLO module. Returns an error if the
  // transform fails, otherwise returns true if any change was made to the
  // module.
  virtual absl::StatusOr<bool> Transform(xla::HloModule* module) = 0;
};

// Registers a user's HloXlaTransform implementation to be applied at the
// specified stage (and associated with the given name). Thread-safe.
void RegisterHloXlaTransform(HloXlaTransform::PipelineStage stage,
                             std::shared_ptr<HloXlaTransform> transform);

// Returns the list of registered HloXlaTransforms for the given stage.
// Thread-safe.
std::vector<std::shared_ptr<HloXlaTransform>> GetHloXlaTransforms(
    HloXlaTransform::PipelineStage stage);

// Clears all registered HloXlaTransforms. Returns true if any transforms were
// cleared, false otherwise.
// Thread-safe.
bool ClearHloXlaTransforms();

// Clears a specific registered HloXlaTransform. Returns true if the transform
// was found and cleared, false otherwise.
// Thread-safe.
bool ClearHloXlaTransform(HloXlaTransform::PipelineStage stage,
                          absl::string_view name);

// Applies all registered HloXlaTransforms for the specified stage to the
// given module. Returns an error if any transform fails, otherwise returns true
// if any transform made a change to the module. Thread-safe.
absl::StatusOr<bool> ApplyXlaTransformsToModule(
    HloXlaTransform::PipelineStage stage, xla::HloModule* module);

// HloPass that applies all registered HloXlaTransforms for the specified stage.
// HloXlaTransforms which are registered at the same stage, are applied in the
// order in which they were registered.
class ApplyXlaTransforms : public HloModulePass {
 public:
  explicit ApplyXlaTransforms(HloXlaTransform::PipelineStage stage)
      : stage_(stage) {}
  ~ApplyXlaTransforms() override = default;

  absl::string_view name() const override { return "apply-xla-transforms"; }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloXlaTransform::PipelineStage stage_;
};

// Replaces the contents of `module` with the HloModule described by
// `transformed_proto`.
absl::Status UpdateHloModuleFromProto(HloModule* module,
                                      const HloModuleProto& transformed_proto);

}  // namespace xla

#endif  // XLA_SERVICE_XLA_TRANSFORM_H_
