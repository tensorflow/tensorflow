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

#ifndef XLA_TESTS_HLO_PJRT_TEST_BASE_H_
#define XLA_TESTS_HLO_PJRT_TEST_BASE_H_

// Inclusion of this header indicates that the test has been migrated to use
// HloRunnerPjRt. Since this requires tagging the build target, the whole target
// must be migrated at once. HloPjRtTestBase cannot be used in conjunction with
// HloTestBase and vice versa. This macro helps to ensure that migration test
// base classes are not used in conjunction with HloTestBase.
// TODO: b/408276009 - Remove these macros once all tests have been migrated.
#define XLA_TEST_MIGRATED_TO_HLO_RUNNER_PJRT
#ifdef XLA_TEST_NOT_MIGRATED_TO_HLO_RUNNER_PJRT
static_assert(false,
              "HloPjRtTestBase cannot be used in the same target as a test "
              "that uses HloTestBase.");
#endif  // XLA_TEST_NOT_MIGRATED_TO_HLO_RUNNER_PJRT

#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct HloPjRtTestBaseOptions {
  bool verifier_layout_sensitive = false;
  bool allow_mixed_precision_in_hlo_verifier = true;
  HloPredicate instruction_can_change_layout_func;
};

class HloPjRtTestBase : public HloRunnerAgnosticTestBase {
 protected:
  // This uses the PjRt interpreter backend for the reference backend and
  // automatically finds a PjRt backend for the test backend.
  explicit HloPjRtTestBase(HloPjRtTestBaseOptions options = {});
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_PJRT_TEST_BASE_H_
