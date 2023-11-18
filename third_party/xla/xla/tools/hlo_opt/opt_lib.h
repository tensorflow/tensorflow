/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TOOLS_HLO_OPT_OPT_LIB_H_
#define XLA_TOOLS_HLO_OPT_OPT_LIB_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/statusor.h"
#include "xla/stream_executor/platform.h"
#include "xla/types.h"

namespace xla {

// Platform-specific provider of `hlo_translate` functionality.
struct OptProvider {
  // Generates textual output for a given stage on a given platform, returns
  // empty optional if the stage is not supported.
  virtual StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view stage) = 0;

  virtual ~OptProvider() = default;

  virtual std::vector<std::string> SupportedStages() = 0;

  static void RegisterForPlatform(
      se::Platform::Id platform,
      std::unique_ptr<OptProvider> translate_provider);

  static OptProvider* ProviderForPlatform(se::Platform::Id platform);
};

}  // namespace xla

#endif  // XLA_TOOLS_HLO_OPT_OPT_LIB_H_
