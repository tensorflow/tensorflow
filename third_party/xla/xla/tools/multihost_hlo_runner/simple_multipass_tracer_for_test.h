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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_SIMPLE_MULTIPASS_TRACER_FOR_TEST_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_SIMPLE_MULTIPASS_TRACER_FOR_TEST_H_

namespace xla {
namespace profiler {

// Registers the SimpleMultiPassTracer factory with the TSL profiler registry.
// SimpleMultiPassTracer coordinates multi-pass profiling for workloads on
// accelerators.
void RegisterSimpleMultiPassTracerForTest();

}  // namespace profiler
}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_SIMPLE_MULTIPASS_TRACER_FOR_TEST_H_
