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

#ifndef XLA_TESTS_HLO_TEST_BASE_H_
#define XLA_TESTS_HLO_TEST_BASE_H_

#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_legacy.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base_legacy.h"
#include "xla/util.h"

// Inclusion of this header indicates that the test has NOT been migrated to use
// HloRunnerPjRt. Migration requires tagging the build target so that the
// correct dependencies are included. The whole target must be migrated at once.
// This macro helps to ensure that migration test base classes are not used in
// conjunction with HloTestBase.
// TODO: b/408276009 - Remove these macros once all tests have been migrated.
#define XLA_TEST_NOT_MIGRATED_TO_HLO_RUNNER_PJRT
#ifdef XLA_TEST_MIGRATED_TO_HLO_RUNNER_PJRT
static_assert(false,
              "HloTestBase cannot be used in the same target as a test that "
              "has been explicitly migrated to use HloRunnerPjRt.");
#endif  // XLA_TEST_MIGRATED_TO_HLO_RUNNER_PJRT

namespace xla {

// A base class for tests which build and/or run HLO code. The class includes
// support for running an HLO module on two platforms and compare the results.
// This is a lower level of abstraction than using the client interface and
// enables, for one, explicitly building a graph of HLO instructions to run.
//
// This can also be used to write text/file-based test cases. Note that the test
// target is responsible for linking the needed backends. A convenient way to do
// this is to make it an xla_test: it will generate test targets linking with
// the respective backends, which will be used as the test backend; the
// interpreter backend is already linked with hlo_test_base so it will be the
// default reference backend. For example, if you want to compare both cpu vs.
// interpreter, and gpu vs. interpreter, you can:
//
//  xla_test (
//    name = "sample_text_test",
//    srcs = ["sample_text_test.cc"],
//    backends = [
//      "cpu",
//      "gpu",
//    ],
//    deps = [
//      "//xla/tests:hlo_test_base",
//      ...
//    ],
//  )
//
// For a more detailed example, see "../tests/sample_text_test.cc".
class ABSL_DEPRECATED(
    "Please avoid introducing new tests that use this class. Tests that use "
    "this base class are being incrementally migrated to use HloPjRtTestBase "
    "or HloRunnerAgnosticTestBase directly.") HloTestBase
    : public HloTestBaseLegacy {
 protected:
  // This uses the interpreter backend as the reference backend and
  // automatically finds another supported backend as the test backend. If the
  // interpreter is the only supported backend, it will be both the test backend
  // and the reference backend.
  explicit HloTestBase(bool verifier_layout_sensitive = false,
                       bool allow_mixed_precision_in_hlo_verifier = true,
                       HloPredicate instruction_can_change_layout_func = {})
      : HloTestBaseLegacy(verifier_layout_sensitive,
                          allow_mixed_precision_in_hlo_verifier,
                          std::move(instruction_can_change_layout_func),
                          DefaultRunnerFactory) {}

  // If your test doesn't use interpreter as the reference backend, you can use
  // this constructor. Note that your test target is responsible for linking in
  // both needed backends.
  HloTestBase(se::Platform* test_platform, se::Platform* reference_platform,
              bool verifier_layout_sensitive = false,
              bool allow_mixed_precision_in_hlo_verifier = true,
              HloPredicate instruction_can_change_layout_func = {})
      : HloTestBaseLegacy(test_platform, reference_platform,
                          verifier_layout_sensitive,
                          allow_mixed_precision_in_hlo_verifier,
                          std::move(instruction_can_change_layout_func),
                          DefaultRunnerFactory) {}

 private:
  static std::unique_ptr<HloRunnerLegacy> DefaultRunnerFactory(
      se::Platform* platform) {
    return std::make_unique<HloRunner>(platform);
  }
};

}  // namespace xla

#endif  // XLA_TESTS_HLO_TEST_BASE_H_
