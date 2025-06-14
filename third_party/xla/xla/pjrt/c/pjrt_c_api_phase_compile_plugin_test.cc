/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/pjrt/c/pjrt_c_api_phase_compile_plugin.h"

#include <gtest/gtest.h>
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_phase_compile_extension.h"

namespace pjrt {

class PhaseCompileTest : public ::testing::Test {
 protected:
  static const PJRT_Api* api_;
  static void SetUpTestSuite() { api_ = GetPjrtApi(); }
  static void TearDownTestSuite() {}
};

const PJRT_Api* PhaseCompileTest::api_ = nullptr;

// Test registration of PhaseCompile extension.
TEST_F(PhaseCompileTest, TestExtensionRegistration) {
  auto phase_compile_extension =
      pjrt::FindExtension<PJRT_PhaseCompile_Extension>(
          api_, PJRT_Extension_Type::PJRT_Extension_Type_PhaseCompile);
  EXPECT_NE(phase_compile_extension, nullptr);
}

}  // namespace pjrt
