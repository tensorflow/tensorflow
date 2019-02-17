/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUSE_MAX_POOL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_PASSES_FUSE_MAX_POOL_H_

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"

namespace xla {

namespace poplarplugin {

// The purpose of this pass is to extract and match forward and backwards
// MaxPools together
class FuseMaxPool : public HloMatcher {
 public:
  FuseMaxPool(struct CompilerAnnotations& annotations);

  ~FuseMaxPool() override = default;

  absl::string_view name() const override { return "poplar-fuse-max-pool"; }

 private:
  bool HandleMatch(HloMatcherMatched& match,
                   const absl::optional<int64> sharding_device) override;

  std::string op_prefix_ = "_pop_op_";

  std::map<const HloInstruction*, HloInstruction*> input_to_fwd_max_pool_map_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
