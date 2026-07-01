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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
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
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cse.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/strings/proto_serialization.h"

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

TEST_F(XlaTransformTest, ClearTransforms) {
  auto axl = std::make_shared<TrivialTransform>("test_transform");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler, axl);

  EXPECT_TRUE(ClearHloXlaTransforms());
  EXPECT_FALSE(ClearHloXlaTransforms());
}

TEST_F(XlaTransformTest, ClearTransform) {
  auto axl1 = std::make_shared<TrivialTransform>("test_transform1");
  auto axl2 = std::make_shared<TrivialTransform>("test_transform2");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler, axl1);
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler, axl2);

  EXPECT_TRUE(ClearHloXlaTransform(
      HloXlaTransform::PipelineStage::kPreScheduler, "test_transform1"));

  const auto& transforms =
      GetHloXlaTransforms(HloXlaTransform::PipelineStage::kPreScheduler);
  ASSERT_EQ(transforms.size(), 1);
  EXPECT_EQ(transforms[0]->name(), "test_transform2");

  EXPECT_FALSE(ClearHloXlaTransform(
      HloXlaTransform::PipelineStage::kPreScheduler, "test_transform1"));
  EXPECT_TRUE(ClearHloXlaTransform(
      HloXlaTransform::PipelineStage::kPreScheduler, "test_transform2"));
  EXPECT_TRUE(GetHloXlaTransforms(HloXlaTransform::PipelineStage::kPreScheduler)
                  .empty());
}

TEST_F(XlaTransformTest, ApplyTransforms) {
  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      ROOT add = f32[] add(p0, p0)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));

  auto transform = std::make_shared<TrivialTransform>("trivial_transform");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler,
                          transform);

  ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(XlaTransformTest, TransformMixedPrecisionPad) {
  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY test_computation {
      p0 = bf16[1,1,4,4] parameter(0)
      p1 = bf16[1,1,4,4] parameter(1)
      c0 = f32[] constant(0)
      ROOT pad.0 = bf16[1,1,8,4] pad(p1, c0), padding=0_0x0_0x0_4x0_0
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module,
                       xla::ParseAndReturnUnverifiedModule(hlo_text));

  auto transform = std::make_shared<TrivialTransform>("trivial_transform");
  RegisterHloXlaTransform(HloXlaTransform::PipelineStage::kPreScheduler,
                          transform);

  HloPassPipeline pipeline("test_pipeline");

  AlgebraicSimplifierOptions options;
  pipeline.AddPass<AlgebraicSimplifier>(options);
  pipeline.AddPass<ApplyXlaTransforms>(
      HloXlaTransform::PipelineStage::kPreScheduler);
  pipeline.AddPass<HloTrivialScheduler>();
  pipeline.AddPass<ApplyXlaTransforms>(
      HloXlaTransform::PipelineStage::kPostScheduler);

  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_text));

  ASSERT_OK_AND_ASSIGN(bool changed, pipeline.Run(module.get()));
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
    ASSERT_OK_AND_ASSIGN(auto config, HloModule::CreateModuleConfigFromProto(
                                          proto, debug_options));
    ASSERT_OK_AND_ASSIGN(auto module,
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

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_text));

  LOG(INFO) << "HloModule before XLA transforms:\n" << module->ToString();

  ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);

  LOG(INFO) << "HloModule after XLA transforms:\n" << module->ToString();

  EXPECT_EQ(module->entry_computation()->root_instruction()->opcode(),
            HloOpcode::kNegate);

  // Clear the transform before callbacks goes out of scope.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "pjrt_c_api_transform";
    args.name_size = sizeof("pjrt_c_api_transform") - 1;
    args.callbacks = nullptr;
    args.cleared = false;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(args.cleared);
  }
}

TEST_F(XlaTransformTest, PjrtCApiExtensionClear) {
  PJRT_Xla_Transform_Extension extension = pjrt::CreateXlaTransformExtension();

  // Register a transform.
  PJRT_XlaTransform_Callbacks callbacks;
  callbacks.version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION;
  callbacks.dtor = nullptr;
  callbacks.transform_hlo_module = [](PJRT_XlaTransform_Callbacks* callbacks,
                                      PJRT_XlaTransform_Args* args) {};

  PJRT_Register_Xla_Transform_Args reg_args;
  reg_args.struct_size = PJRT_Register_Xla_Transform_Args_STRUCT_SIZE;
  reg_args.name = "pjrt_c_api_transform";
  reg_args.name_size = sizeof("pjrt_c_api_transform") - 1;
  reg_args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
  reg_args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_xla_transform(&reg_args);
  EXPECT_EQ(error, nullptr);

  // Clear with wrong name should return false.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "wrong_name";
    args.name_size = sizeof("wrong_name") - 1;
    args.callbacks = nullptr;
    args.cleared = true;  // initialize to true to make sure it changes
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_FALSE(args.cleared);
  }

  // Clear with correct name should return true.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "pjrt_c_api_transform";
    args.name_size = sizeof("pjrt_c_api_transform") - 1;
    args.callbacks = nullptr;
    args.cleared = false;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(args.cleared);
  }

  // Clear again should return false.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "pjrt_c_api_transform";
    args.name_size = sizeof("pjrt_c_api_transform") - 1;
    args.callbacks = nullptr;
    args.cleared = true;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_FALSE(args.cleared);
  }
}

TEST_F(XlaTransformTest, PjrtCApiExtensionClearByCallbacks) {
  PJRT_Xla_Transform_Extension extension = pjrt::CreateXlaTransformExtension();

  // Register a transform without a name.
  PJRT_XlaTransform_Callbacks callbacks;
  callbacks.version = PJRT_API_XLA_TRANSFORM_EXTENSION_VERSION;
  callbacks.dtor = nullptr;
  callbacks.transform_hlo_module = [](PJRT_XlaTransform_Callbacks* callbacks,
                                      PJRT_XlaTransform_Args* args) {};

  PJRT_Register_Xla_Transform_Args reg_args;
  reg_args.struct_size = PJRT_Register_Xla_Transform_Args_STRUCT_SIZE;
  reg_args.name = nullptr;
  reg_args.name_size = 0;
  reg_args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
  reg_args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_xla_transform(&reg_args);
  EXPECT_EQ(error, nullptr);

  // Clear with correct callbacks should return true.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = nullptr;
    args.name_size = 0;
    args.callbacks = &callbacks;
    args.cleared = false;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(args.cleared);
  }
}

TEST_F(XlaTransformTest, PjrtCApiExtensionPreservesSchedule) {
  // 1. Define the C callback.
  auto c_callback = [](PJRT_XlaTransform_Callbacks* callbacks,
                       PJRT_XlaTransform_Args* args) {
    EXPECT_NE(args->hlo_module.data, nullptr);
    EXPECT_GT(args->hlo_module.size, 0);

    xla::HloModuleProto proto;
    EXPECT_TRUE(proto.ParseFromString(
        absl::string_view(args->hlo_module.data, args->hlo_module.size)));

    DebugOptions debug_options;
    ASSERT_OK_AND_ASSIGN(auto config, HloModule::CreateModuleConfigFromProto(
                                          proto, debug_options));
    ASSERT_OK_AND_ASSIGN(auto module,
                         HloModule::CreateFromProto(proto, config));

    // Return it as is but mark as changed.
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
  args.name = "pjrt_c_api_transform_schedule";
  args.name_size = sizeof("pjrt_c_api_transform_schedule") - 1;
  args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
  args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_xla_transform(&args);
  EXPECT_EQ(error, nullptr);

  absl::string_view hlo_text = R"(
    HloModule test_module, is_scheduled=true
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      add = f32[] add(p0, p0)
      neg = f32[] negate(p0)
      ROOT tuple = (f32[], f32[]) tuple(add, neg)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_text));

  EXPECT_TRUE(module->has_schedule());
  std::vector<std::string> expected_order;
  for (const HloInstruction* inst : module->schedule()
                                        .sequence(module->entry_computation())
                                        .instructions()) {
    expected_order.push_back(std::string(inst->name()));
  }

  ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);

  // Verify that it still has a schedule.
  EXPECT_TRUE(module->has_schedule());
  LOG(INFO) << "Schedule after transform:\n" << module->schedule().ToString();
  EXPECT_FALSE(module->schedule().empty());
  auto status = module->schedule().Verify();
  LOG(INFO) << "Schedule verify status: " << status;
  EXPECT_TRUE(status.ok());

  std::vector<std::string> actual_order;
  for (const HloInstruction* inst : module->schedule()
                                        .sequence(module->entry_computation())
                                        .instructions()) {
    actual_order.push_back(std::string(inst->name()));
  }
  EXPECT_EQ(expected_order, actual_order);

  // Clear the transform before callbacks goes out of scope.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "pjrt_c_api_transform_schedule";
    args.name_size = sizeof("pjrt_c_api_transform_schedule") - 1;
    args.callbacks = nullptr;
    args.cleared = false;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(args.cleared);
  }
}

TEST_F(XlaTransformTest, PjrtCApiExtensionUnusedComputationScheduleUAF) {
  // 1. Define the C callback.
  auto c_callback = [](PJRT_XlaTransform_Callbacks* callbacks,
                       PJRT_XlaTransform_Args* args) {
    EXPECT_NE(args->hlo_module.data, nullptr);
    EXPECT_GT(args->hlo_module.size, 0);

    xla::HloModuleProto proto;
    EXPECT_TRUE(proto.ParseFromString(
        absl::string_view(args->hlo_module.data, args->hlo_module.size)));

    DebugOptions debug_options;
    ASSERT_OK_AND_ASSIGN(auto config, HloModule::CreateModuleConfigFromProto(
                                          proto, debug_options));
    ASSERT_OK_AND_ASSIGN(auto module,
                         HloModule::CreateFromProto(proto, config));

    // Add unused computation
    auto builder = HloComputation::Builder("unused_comp");
    builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "p0"));
    auto unused_comp = module->AddEmbeddedComputation(builder.Build());
    if (module->has_schedule()) {
      module->schedule().GetOrCreateSequence(unused_comp);
    }

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
  args.name = "pjrt_c_api_transform_uaf";
  args.name_size = sizeof("pjrt_c_api_transform_uaf") - 1;
  args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
  args.callbacks = &callbacks;

  PJRT_Error* error = extension.register_xla_transform(&args);
  EXPECT_EQ(error, nullptr);

  absl::string_view hlo_text = R"(
    HloModule test_module, is_scheduled=true
    ENTRY test_computation {
      p0 = f32[] parameter(0)
      add = f32[] add(p0, p0)
      neg = f32[] negate(p0)
      ROOT tuple = (f32[], f32[]) tuple(add, neg)
    }
  )";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo_text));

  EXPECT_TRUE(module->has_schedule());

  // This should not crash.
  ASSERT_OK_AND_ASSIGN(
      bool changed,
      ApplyXlaTransformsToModule(HloXlaTransform::PipelineStage::kPreScheduler,
                                 module.get()));
  EXPECT_TRUE(changed);

  // Verify that it still has a schedule and it doesn't contain the unused comp.
  EXPECT_TRUE(module->has_schedule());
  auto status = module->schedule().Verify();
  EXPECT_TRUE(status.ok());

  // Clear the transform before callbacks goes out of scope.
  {
    PJRT_Clear_Xla_Transform_Args args;
    args.struct_size = PJRT_Clear_Xla_Transform_Args_STRUCT_SIZE;
    args.stage = PJRT_XlaTransform_PipelineStage_kPreScheduler;
    args.name = "pjrt_c_api_transform_uaf";
    args.name_size = sizeof("pjrt_c_api_transform_uaf") - 1;
    args.callbacks = nullptr;
    args.cleared = false;
    PJRT_Error* error = extension.clear_xla_transform(&args);
    EXPECT_EQ(error, nullptr);
    EXPECT_TRUE(args.cleared);
  }
}

class UpdateHloModuleFromProtoTest : public XlaTransformTest {
 protected:
  absl::StatusOr<std::unique_ptr<HloModule>>
  CreateModuleWithNonDefaultLayout() {
    absl::string_view hlo_text = R"(
      HloModule test_module
      ENTRY main {
        p0 = f32[2,3] parameter(0)
        ROOT neg = f32[2,3] negate(p0)
      }
    )";
    ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo_text));
    Shape non_default_shape = NonDefaultShape();
    *module->mutable_entry_computation_layout()->mutable_parameter_layout(0) =
        ShapeLayout(non_default_shape);
    *module->mutable_entry_computation_layout()->mutable_result_layout() =
        ShapeLayout(non_default_shape);
    return module;
  }

  absl::StatusOr<HloModuleProto> HloTextToProto(absl::string_view hlo_text) {
    ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(hlo_text));
    return module->ToProto();
  }

  Shape NonDefaultShape() {
    return ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  }
};

TEST_F(UpdateHloModuleFromProtoTest, PropagatesNewLayout) {
  ASSERT_OK_AND_ASSIGN(auto module, CreateModuleWithNonDefaultLayout());

  absl::string_view hlo_text = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2,3] parameter(0)
      ROOT neg = f32[2,3] negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(HloModuleProto transformed_proto,
                       HloTextToProto(hlo_text));

  EXPECT_TRUE(UpdateHloModuleFromProto(module.get(), transformed_proto).ok());

  // The layout should be updated to the default layout from the proto,
  // instead of preserving the original non-default layout.
  Shape default_shape = ShapeUtil::MakeShape(F32, {2, 3});
  EXPECT_EQ(module->entry_computation_layout().parameter_layout(0),
            ShapeLayout(default_shape));
  EXPECT_EQ(module->entry_computation_layout().result_layout(),
            ShapeLayout(default_shape));
}

TEST_F(UpdateHloModuleFromProtoTest, PropagatesNonDefaultLayout) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2,3] parameter(0)
      ROOT neg = f32[2,3] negate(p0)
    }
  )"));

  absl::string_view transformed_hlo = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2,3]{0,1} parameter(0)
      ROOT neg = f32[2,3]{0,1} negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(auto transformed_module,
                       ParseAndReturnUnverifiedModule(transformed_hlo));
  HloModuleProto transformed_proto = transformed_module->ToProto();

  EXPECT_TRUE(UpdateHloModuleFromProto(module.get(), transformed_proto).ok());

  // The layout should be updated to the non-default layout from the proto.
  EXPECT_EQ(module->entry_computation_layout().parameter_layout(0),
            ShapeLayout(NonDefaultShape()));
  EXPECT_EQ(module->entry_computation_layout().result_layout(),
            ShapeLayout(NonDefaultShape()));
}

TEST_F(UpdateHloModuleFromProtoTest, IncompatibleShape) {
  ASSERT_OK_AND_ASSIGN(auto module, CreateModuleWithNonDefaultLayout());

  absl::string_view transformed_hlo_text = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[3,4] parameter(0)
      ROOT neg = f32[3,4] negate(p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(HloModuleProto transformed_proto,
                       HloTextToProto(transformed_hlo_text));

  EXPECT_FALSE(UpdateHloModuleFromProto(module.get(), transformed_proto).ok());

  EXPECT_EQ(module->entry_computation_layout().parameter_layout(0),
            ShapeLayout(NonDefaultShape()));
  EXPECT_EQ(module->entry_computation_layout().result_layout(),
            ShapeLayout(NonDefaultShape()));
}

TEST_F(UpdateHloModuleFromProtoTest, IncompatibleParamCount) {
  ASSERT_OK_AND_ASSIGN(auto module, CreateModuleWithNonDefaultLayout());

  absl::string_view transformed_hlo_text = R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2,3] parameter(0)
      p1 = f32[2,3] parameter(1)
      ROOT add = f32[2,3] add(p0, p1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(HloModuleProto transformed_proto,
                       HloTextToProto(transformed_hlo_text));

  EXPECT_FALSE(UpdateHloModuleFromProto(module.get(), transformed_proto).ok());

  EXPECT_EQ(module->entry_computation_layout().parameter_layout(0),
            ShapeLayout(NonDefaultShape()));
}

TEST_F(UpdateHloModuleFromProtoTest, PreservesAliasAndDonorConfigs) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test_module
    ENTRY main {
      p0 = f32[2,3] parameter(0)
      ROOT neg = f32[2,3] negate(p0)
    }
  )"));

  HloInputOutputAliasConfig alias_config(
      module->entry_computation()->root_instruction()->shape());
  ASSERT_OK(alias_config.SetUpAlias({}, 0, {}));
  module->set_input_output_alias_config(alias_config);

  HloBufferDonorConfig donor_config;
  ASSERT_OK(donor_config.AddBufferDonor(0, {}));
  module->set_buffer_donor_config(donor_config);

  HloModuleProto transformed_proto = module->ToProto();

  EXPECT_OK(UpdateHloModuleFromProto(module.get(), transformed_proto));

  EXPECT_TRUE(module->input_output_alias_config().ParameterHasAlias(0, {}));
  EXPECT_TRUE(module->buffer_donor_config().ParameterIsBufferDonor(0, {}));
}

}  // namespace

}  // namespace xla
