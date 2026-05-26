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

#include "xla/service/xla_transform.h"

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/hlo/transforms/simplifiers/hlo_memory_scheduler.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_extension.h"
#include "xla/pjrt/c/pjrt_c_api_xla_transform_internal.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cse.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

namespace {

class TrivialTransform : public HloXlaTransform {
 public:
  explicit TrivialTransform(std::string name)
      : HloXlaTransform(std::move(name)) {}
  absl::StatusOr<bool> Transform(xla::HloModule* module) override {
    return true;
  }
};

class XlaTransformTest : public ::testing::Test {
 protected:
  void SetUp() override { ClearHloXlaTransforms(); }
  void TearDown() override { ClearHloXlaTransforms(); }
};

TEST_F(XlaTransformTest, Registration) {
  auto axl = std::make_shared<TrivialTransform>("test_transform");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler, axl);

  const auto& transforms =
      GetHloXlaTransforms(HloXlaTransform::PipelineStage::kPreScheduler);
  ASSERT_EQ(transforms.size(), 1);
  EXPECT_EQ(transforms[0]->name(), "test_transform");
  EXPECT_NE(transforms[0], nullptr);
}

TEST_F(XlaTransformTest, ApplyTransforms) {
  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      ROOT add = f32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          xla::ParseAndReturnUnverifiedModule(hlo_text));

  auto transform = std::make_shared<TrivialTransform>("trivial_transform");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler,
                          transform);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(XlaTransformTest, PassPipeline) {
  auto pre_transform = std::make_shared<TrivialTransform>("pre_trivial");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler,
                          pre_transform);

  auto post_transform = std::make_shared<TrivialTransform>("post_trivial");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPostScheduler,
                          post_transform);

  HloPassPipeline pipeline("test_pipeline");

  AlgebraicSimplifierOptions options;
  pipeline.AddPass<AlgebraicSimplifier>(options);
  pipeline.AddPass<ApplyXlaTransforms>(
      HloXlaTransform::PipelineStage::kPreScheduler);
  pipeline.AddPass<HloTrivialScheduler>();
  pipeline.AddPass<ApplyXlaTransforms>(
      HloXlaTransform::PipelineStage::kPostScheduler);
  pipeline.AddPass<HloCSE>(/*is_layout_sensitive=*/false);

  const char* hlo_text = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT neg = f32[] negate(p0)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_text));

  TF_ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(XlaTransformTest, PjrtCApiExtension) {
  // 1. Define the C callback.
  auto c_callback = [](PJRT_XlaTransform_Callbacks* callbacks,
                       PJRT_XlaTransform_Args* args) {
    EXPECT_NE(args->hlo_module.data, nullptr);
    EXPECT_GT(args->hlo_module.size, 0);

    xla::HloModuleProto proto;
    EXPECT_TRUE(proto.ParseFromString(
        absl::string_view(args->hlo_module.data, args->hlo_module.size)));

    DebugOptions debug_options;
    TF_ASSERT_OK_AND_ASSIGN(auto config, HloModule::CreateModuleConfigFromProto(
                                             proto, debug_options));
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            HloModule::CreateFromProto(proto, config));

    // Modify the module by adding a Negate instruction.
    auto* root = module->entry_computation()->root_instruction();
    auto* negate = module->entry_computation()->AddInstruction(
        HloInstruction::CreateUnary(root->shape(), HloOpcode::kNegate, root));
    module->entry_computation()->set_root_instruction(negate);

    xla::HloModuleProto modified_proto = module->ToProto();
    static thread_local std::string persistent_proto;
    persistent_proto.clear();
    EXPECT_TRUE(
        tsl::SerializeToStringDeterministic(modified_proto, &persistent_proto));

    args->changed = true;
    args->transformed_hlo_module.data = persistent_proto.data();
    args->transformed_hlo_module.size = persistent_proto.size();

    args->header.has_error = false;
  };

  // 2. Create the PJRT_XlaTransform_Callbacks struct.
  PJRT_XlaTransform_Callbacks callbacks;
  callbacks.version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION;
  callbacks.dtor = nullptr;
  callbacks.transform_hlo_module = c_callback;

  // 3. Create the XLA transform extension.
  PJRT_Xla_Transform_Extension extension = pjrt::CreateXlaTransformExtension();

  // 4. Register the transform using the extension.
  PJRT_Register_Xla_Transform_Args args;
  args.struct_size = PJRT_Register_Xla_Transform_Args_STRUCT_SIZE;
  args.name = "pjrt_c_api_transform";
  args.name_size = sizeof("pjrt_c_api_transform") - 1;
  args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
  args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_xla_transform(&args);
  EXPECT_EQ(error, nullptr);

  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      ROOT add = f32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_text));

  LOG(INFO) << "HloModule before XLA transforms:\n" << module->ToString();

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);

  LOG(INFO) << "HloModule after XLA transforms:\n" << module->ToString();

  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kNegate);
}

}  // namespace

}  // namespace xla
