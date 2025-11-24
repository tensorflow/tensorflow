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

#ifndef XLA_HLO_SEPARATE_COMPILATION_HLO_LINKING_MANIFEST_H_
#define XLA_HLO_SEPARATE_COMPILATION_HLO_LINKING_MANIFEST_H_

#include <memory>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/hlo_module_config.h"

namespace xla::separate_compilation {

// Metadata to guide linking process of HLO modules.
//
// Manifest contains Caller/Callee information:
// When splitting modules, sometimes callers and their callees end up
// in different sub-modules. During linking, we must be able to find
// the callees and replace the stubs we inserted in the caller's sub-module.
// Because, HLO serialization includes ids keeping this information in
// the module interferes with caching of artifacts, by making artifacts
// with the same semantics appear different when serialized. This class
// externalizes the information.
struct HloLinkingManifest {
  // Maps from a stub computation to the cloned computation.
  // Note that these `HloComputation` pointers might be from different modules.
  absl::flat_hash_map<const HloComputation* absl_nonnull,
                      const HloComputation* absl_nonnull>
      stub_links;
  std::shared_ptr<const HloModuleConfig> module_config;
  std::unique_ptr<CompilationEnvironments> compilation_environment;
};

}  // namespace xla::separate_compilation
#endif  // XLA_HLO_SEPARATE_COMPILATION_HLO_LINKING_MANIFEST_H_
