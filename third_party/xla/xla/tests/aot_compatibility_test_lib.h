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
#ifndef XLA_TESTS_AOT_COMPATIBILITY_TEST_LIB_H_
#define XLA_TESTS_AOT_COMPATIBILITY_TEST_LIB_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xla/pjrt/pjrt_client.h"
#include "xla/shape.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"

namespace xla {

// Configuration for an AOT compatibility test instance.
struct AotTestConfig {
  bool is_golden;
  int version;
};

// Functor for generating the test suffix in INSTANTIATE_TEST_SUITE_P.
struct AotTestConfigNameGenerator {
  std::string operator()(
      const ::testing::TestParamInfo<AotTestConfig>& info) const;
};

// Discovers the available AOT versions by scanning the provided runfiles
// directory. Looks for files matching "v*.pbtxt".
// Returns a configuration for Golden, plus configs for all found versions.
std::vector<AotTestConfig> GetAvailableAotVersions(const std::string& test_dir);

// A base class for AOT backward/forward compatibility testing.
// This test fixture intercepts PjRtClient compilation and loading
// calls and delegates them to the `AOTInterceptionPjrtClient`.
// The mode (golden vs version) and version number are parameterized.
class AotCompatibilityTestBase
    : public HloRunnerAgnosticTestBase,
      public ::testing::WithParamInterface<AotTestConfig> {
 public:
  struct ClientData {
    std::unique_ptr<PjRtClient> client;
    std::function<Shape(const Shape&)> rep_fn;
    std::function<int64_t(const Shape&)> size_fn;
  };

  explicit AotCompatibilityTestBase(std::string artifact_dir);
  ~AotCompatibilityTestBase() override = default;

 private:
  explicit AotCompatibilityTestBase(ClientData data);
};

}  // namespace xla

#endif  // XLA_TESTS_AOT_COMPATIBILITY_TEST_LIB_H_
