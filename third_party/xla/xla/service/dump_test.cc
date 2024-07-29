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

#include "xla/service/dump.h"

#include <memory>
#include <string>

#include "absl/strings/match.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_parser.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(DumpHloIfEnabled, LargeConstantElided) {
  HloModuleConfig config;
  DebugOptions options = config.debug_options();
  auto env = tsl::Env::Default();
  std::string dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_dump_hlo_as_text(true);
  options.set_xla_dump_large_constants(false);
  config.set_debug_options(options);
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  std::string dump_name = "dump";
  auto paths = DumpHloModuleIfEnabled(*m, dump_name);
  EXPECT_EQ(paths.size(), 1);
  std::string data;
  EXPECT_TRUE(ReadFileToString(env, paths[0], &data).ok());
  EXPECT_TRUE(absl::StrContains(data, "{...}"));
}

TEST(DumpHloIfEnabled, LargeConstantPrinted) {
  HloModuleConfig config;
  DebugOptions options = config.debug_options();
  auto env = tsl::Env::Default();
  std::string dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_dump_hlo_as_text(true);
  options.set_xla_dump_large_constants(true);
  config.set_debug_options(options);
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  std::string dump_name = "dump";
  auto paths = DumpHloModuleIfEnabled(*m, dump_name);
  EXPECT_EQ(paths.size(), 1);
  std::string data;
  EXPECT_TRUE(ReadFileToString(env, paths[0], &data).ok());
  EXPECT_TRUE(!absl::StrContains(data, "{...}"));
}

}  // namespace
}  // namespace xla
