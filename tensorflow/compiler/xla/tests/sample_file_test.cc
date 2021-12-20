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

// This demonstrates how to use hlo_test_base to create a file based testcase
// and compare results on gpu and cpu.

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class SampleFileTest : public HloTestBase {
 protected:
  SampleFileTest()
      : HloTestBase(
            /*test_platform=*/PlatformUtil::GetPlatform("gpu").ValueOrDie(),
            /*reference_platform=*/PlatformUtil::GetPlatform("cpu")
                .ValueOrDie()) {}
};

TEST_F(SampleFileTest, Convolution) {
  const std::string& filename = tensorflow::GetDataDependencyFilepath(
      tensorflow::io::JoinPath("tensorflow", "compiler", "xla", "tests",
                               "isolated_convolution.hlo"));
  EXPECT_TRUE(RunAndCompareFromFile(filename, ErrorSpec{0.01}));
}

}  // namespace
}  // namespace xla
