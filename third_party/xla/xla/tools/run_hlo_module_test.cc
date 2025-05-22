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

#include "xla/tools/run_hlo_module.h"

#include <string>

#include <gtest/gtest.h>
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tools/run_hlo_module.pb.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

RunHloModuleIterationLiterals GetTestProto() {
  RunHloModuleIterationLiterals result;
  *result.add_arguments() = LiteralUtil::CreateR1<float>({0.1, 0.2}).ToProto();
  *result.add_arguments() = LiteralUtil::CreateR1<float>({0.3, 0.4}).ToProto();
  *result.mutable_result() = LiteralUtil::CreateR1<float>({0.5, 0.6}).ToProto();
  *result.mutable_reference_result() =
      LiteralUtil::CreateR1<float>({0.5, 0.6}).ToProto();
  return result;
}

TEST(ReadInputLiteralsFromFile, ReadRunHloModuleLiteralsBinaryProto) {
  std::string file_path;
  auto env = tsl::Env::Default();
  EXPECT_TRUE(env->LocalTempFilename(&file_path));
  auto proto = GetTestProto();
  RunHloModuleLiterals wrapped_proto;
  *wrapped_proto.add_iterations() = proto;
  TF_ASSERT_OK(tsl::WriteBinaryProto(env, file_path, wrapped_proto));
  RunHloModuleLiterals result;
  ReadInputLiteralsFromFile(file_path, &result);
  EXPECT_EQ(result.SerializeAsString(), wrapped_proto.SerializeAsString());
}

TEST(ReadInputLiteralsFromFile, ReadRunHloModuleLiteralsTextProto) {
  std::string file_path;
  auto env = tsl::Env::Default();
  EXPECT_TRUE(env->LocalTempFilename(&file_path));
  auto proto = GetTestProto();
  RunHloModuleLiterals wrapped_proto;
  *wrapped_proto.add_iterations() = proto;
  TF_ASSERT_OK(tsl::WriteTextProto(env, file_path, wrapped_proto));
  RunHloModuleLiterals result;
  ReadInputLiteralsFromFile(file_path, &result);
  EXPECT_EQ(result.SerializeAsString(), wrapped_proto.SerializeAsString());
}

TEST(ReadInputLiteralsFromFile, ReadRunHloModuleIterationLiteralsBinaryProto) {
  std::string file_path;
  auto env = tsl::Env::Default();
  EXPECT_TRUE(env->LocalTempFilename(&file_path));
  auto proto = GetTestProto();
  TF_ASSERT_OK(tsl::WriteBinaryProto(env, file_path, proto));
  RunHloModuleLiterals result;
  ReadInputLiteralsFromFile(file_path, &result);
  EXPECT_EQ(result.iterations_size(), 1);
  EXPECT_EQ(result.iterations(0).SerializeAsString(),
            proto.SerializeAsString());
}

TEST(ReadInputLiteralsFromFile, ReadRunHloModuleIterationLiteralsTextProto) {
  std::string file_path;
  auto env = tsl::Env::Default();
  EXPECT_TRUE(env->LocalTempFilename(&file_path));
  auto proto = GetTestProto();
  TF_ASSERT_OK(tsl::WriteTextProto(env, file_path, proto));
  RunHloModuleLiterals result;
  ReadInputLiteralsFromFile(file_path, &result);
  EXPECT_EQ(result.iterations_size(), 1);
  EXPECT_EQ(result.iterations(0).SerializeAsString(),
            proto.SerializeAsString());
}

}  // namespace
}  // namespace xla
