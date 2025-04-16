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
#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_CONFIG_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_CONFIG_H_

namespace xla {
namespace cpu {

// Whether CPU Fusion emitters are enabled is controlled via XLA_FLAGS.
// Here we define some flags to enable/disable specific emitters. This
// will avoid churn until all emitters are stable.
inline constexpr bool kFusionEmitterScatterEnabled = true;

}  // namespace cpu
}  // namespace xla

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_CPU_FUSION_EMITTER_CONFIG_H_
