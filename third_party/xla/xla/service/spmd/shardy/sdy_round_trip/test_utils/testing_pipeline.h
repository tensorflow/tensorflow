/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_TESTING_PIPELINE_H_
#define XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_TESTING_PIPELINE_H_

namespace xla {
namespace sdy {

// Register the xla-sdy-round-trip-testing-pipeline.
// This takes an SDY module, exports it to MHLO while saving the SDY attrs
// and meshes, goes to HLO, back to MHLO, and then back to SDY.
// This is for testing roundtripping SDY modules, but should be eventually
// removed as part of b/335666088.
void registerSdyRoundTripTestingPipeline();

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_SDY_ROUND_TRIP_TEST_UTILS_TESTING_PIPELINE_H_
