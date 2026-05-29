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

#ifndef XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_TEST_BASE_H_
#define XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_TEST_BASE_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tools/hlo_isolation/hlo_isolation_api.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace hlo_isolation {

template <typename T>
class HloIsolationTestMixin : public T {
 protected:
  template <typename... Args>
  explicit HloIsolationTestMixin(Args&&... args)
      : T(std::forward<Args>(args)...) {}

  void RunAndVerifyIsolationTest(const HloModule& module,
                                 const PipelineIsolationOptions& options = {}) {
    VerifyResults(RunIsolationPipeline(module, &this->test_runner(),
                                       &this->reference_runner(), options));
  }

  void RunAndVerifyIsolationTest(const std::string& input_path,
                                 const PipelineIsolationOptions& options = {}) {
    VerifyResults(RunIsolationPipeline(input_path, &this->test_runner(),
                                       &this->reference_runner(), options));
  }

 private:
  void VerifyResults(
      const absl::StatusOr<std::vector<HloIsolationTestResult>>& results_or) {
    TF_ASSERT_OK(results_or.status());
    for (const HloIsolationTestResult& result : results_or.value()) {
      EXPECT_TRUE(result.state() == SUCCESS || result.state() == SKIPPED)
          << "Isolation test failed for submodule: " << result.module_name()
          << " Reason: " << result.reason();
      if (result.state() == SKIPPED) {
        LOG(INFO) << "Isolation test skipped for submodule: "
                  << result.module_name() << " Reason: " << result.reason();
      }
    }
  }
};

}  // namespace hlo_isolation
}  // namespace xla

#endif  // XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_TEST_BASE_H_
