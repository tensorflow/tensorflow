/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/cpu_gpu_shape_verifier.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/hlo_parser.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

using CpuGpuShapeVerifierTest = HloTestBase;
using ::testing::HasSubstr;

TEST_F(CpuGpuShapeVerifierTest, Int4NotSupported) {
  const char* const hlo_string = R"(
  HloModule Module

  ENTRY main {
    p0 = u4[10] parameter(0)
    ROOT out = u8[10] convert(p0)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));

  HloVerifierOpts opts;
  std::unique_ptr<TargetVerifierMetadata> metadata =
      std::make_unique<CpuGpuVerifierMetadata>(std::move(opts));
  HloVerifier hlo_verifier(std::move(metadata));
  auto status = hlo_verifier.Run(module.get()).status();
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              HasSubstr("S4/U4 is currently not support on XLA CPU/GPU"));
}

}  // namespace
}  // namespace xla
