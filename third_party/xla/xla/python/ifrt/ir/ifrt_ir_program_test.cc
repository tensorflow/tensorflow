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

#include "xla/python/ifrt/ir/ifrt_ir_program.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/client/executable_build_options.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_compile_options.pb.h"
#include "xla/service/computation_placer.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

TEST(IfrtIRCompileOptionsTest, ToFromProto) {
  IfrtIrCompileOptionsProto proto;
  int num_devices = 8;
  for (int i = 0; i < num_devices; ++i) {
    proto.add_device_ids(i);
  }
  for (int i = 0; i < 4; ++i) {
    xla::CompileOptions src;
    xla::ExecutableBuildOptions build_option;
    build_option.set_device_assignment(xla::DeviceAssignment(2, 4));
    src.executable_build_options = build_option;
    TF_ASSERT_OK_AND_ASSIGN(CompileOptionsProto compile_options_proto,
                            src.ToProto());
    proto.mutable_compile_option_overrides()->insert(
        {absl::StrCat("key", i), compile_options_proto});
  }

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<IfrtIRCompileOptions> options,
                          IfrtIRCompileOptions::FromProto(proto));

  EXPECT_EQ(options->compile_options_overrides->size(), 4);
  EXPECT_EQ(options->device_assignments.size(), num_devices);
  TF_ASSERT_OK_AND_ASSIGN(IfrtIrCompileOptionsProto from_to_proto,
                          options->ToProto());

  for (int i = 0; i < 4; ++i) {
    std::string key = absl::StrCat("key", i);
    EXPECT_EQ(
        from_to_proto.compile_option_overrides().at(key).SerializeAsString(),
        proto.compile_option_overrides().at(key).SerializeAsString());
  }
  EXPECT_EQ(from_to_proto.compile_option_overrides_size(),
            proto.compile_option_overrides_size());
  EXPECT_EQ(std::vector(from_to_proto.device_ids().begin(),
                        from_to_proto.device_ids().end()),
            std::vector(proto.device_ids().begin(), proto.device_ids().end()));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
