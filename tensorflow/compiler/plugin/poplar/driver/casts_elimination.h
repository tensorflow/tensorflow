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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CASTS_ELIMINATION_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_CASTS_ELIMINATION_H_

#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"
#include "tensorflow/core/framework/types.h"

namespace xla {

namespace poplarplugin {

// The purpose of this pass is to remove any casts we deem unnecessary
class CastsElimination : public HloMatcher {
 public:
  CastsElimination(struct CompilerAnnotations& annotations);

  ~CastsElimination() override = default;

  absl::string_view name() const override { return "poplar-casts-elimination"; }

 private:
  bool HandleMatch(HloMatcherMatched& match,
                   const absl::optional<int64>) override;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
