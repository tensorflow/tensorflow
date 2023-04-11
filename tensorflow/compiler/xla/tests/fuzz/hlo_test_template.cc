/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class HloTest : public HloTestBase {};

TEST_F(HloTest, HLO_TEST_NAME) {
  std::string path_to_hlo =
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tests", "fuzz", HLO_PATH);
  std::string hlo;
  TF_CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), path_to_hlo, &hlo));
  std::cerr << hlo << std::endl;
  HloModuleConfig config;

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo, config));
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01, 0.01}));
}

}  // namespace
}  // namespace xla
