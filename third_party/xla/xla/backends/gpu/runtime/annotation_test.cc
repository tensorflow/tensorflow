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

#include "xla/backends/gpu/runtime/annotation.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "tsl/profiler/lib/nvtx_utils.h"

namespace xla::gpu {
namespace {

using ::testing::HasSubstr;
using ::testing::Not;

TEST(AnnotationTest, UniqueId) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test

    ENTRY main {
      dummy = f32[] constant(0)
      ROOT constant = f32[] constant(1)
    }
  )"));

  const HloInstruction* constant =
      module->entry_computation()->GetInstructionWithName("constant");
  ASSERT_NE(constant, nullptr);

  ModuleAnnotation module_annotation(*module);
  InstructionAnnotation instruction_annotation(module_annotation, *constant);

  auto xprof_name = instruction_annotation.xprof_name();
  EXPECT_EQ(constant->unique_id(), 1);
  EXPECT_THAT(xprof_name, HasSubstr(absl::StrCat("unique_hlo_op_id=",
                                                 constant->unique_id())));
}

TEST(AnnotationTest, CollectiveMetadata) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test, replica_count=4

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main {
      input = f32[8] parameter(0)
      ROOT all_reduce = f32[8] all-reduce(input),
          replica_groups={{0,1},{2,3}}, to_apply=add,
          metadata={op_type="framework.all_reduce", op_name="model/all_reduce",
                    source_file="collectives.py", source_line=23},
          frontend_attributes={collective_group_key="group-0",
                               combiner_key="combine-0",
                               _scheduling_group_id="7",
                               _xla_stream_annotation="collective"},
          backend_config={"collective_backend_config":
              {"is_pipelined":true,"is_spmd_generated":true}}
    }
  )"));

  HloInstruction* input =
      module->entry_computation()->GetInstructionWithName("input");
  HloInstruction* all_reduce = module->entry_computation()->root_instruction();
  ASSERT_NE(input, nullptr);

  ModuleAnnotation module_annotation(*module);
  InstructionAnnotation basic(module_annotation, *all_reduce);
  InstructionAnnotation detailed(module_annotation, *input,
                                 TraceAnnotationLevel::kDetailed);
  InstructionAnnotation collective(module_annotation, *all_reduce,
                                   TraceAnnotationLevel::kDetailed);

  EXPECT_FALSE(basic.has_detailed_annotations());
  EXPECT_TRUE(detailed.has_detailed_annotations());
  EXPECT_FALSE(detailed.is_collective_annotation());
  EXPECT_TRUE(collective.has_detailed_annotations());
  EXPECT_TRUE(collective.is_collective_annotation());

  auto basic_xprof_name = basic.xprof_name();
  auto detailed_xprof_name = detailed.xprof_name();
  auto collective_nvtx_name = collective.nvtx_name();
  auto collective_xprof_name = collective.xprof_name();

  EXPECT_THAT(basic_xprof_name, Not(HasSubstr("shape=")));
  EXPECT_THAT(basic_xprof_name, Not(HasSubstr("replica_groups=")));
  EXPECT_THAT(detailed_xprof_name, HasSubstr("shape=f32[8]"));
  EXPECT_THAT(collective_nvtx_name, Not(HasSubstr("shape=")));
  EXPECT_THAT(collective_nvtx_name, Not(HasSubstr("replica_groups=")));

  EXPECT_THAT(collective_xprof_name, HasSubstr("op_type=framework.all_reduce"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("op_name=model/all_reduce"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("source_file=collectives.py"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("source_line=23"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("shape=f32[8]"));

  EXPECT_THAT(collective_xprof_name, HasSubstr("replica_groups={{0;1};{2;3}}"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("is_pipelined=1"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("is_spmd_generated=1"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("collective_group_key=group-0"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("combiner_key=combine-0"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("scheduling_group_id=7"));
  EXPECT_THAT(collective_xprof_name, HasSubstr("stream_annotation=collective"));
}

TEST(AnnotationTest, AsyncCollectiveMetadata) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test, replica_count=4

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT sum = f32[] add(lhs, rhs)
    }

    ENTRY main {
      input = f32[8] parameter(0)
      all_reduce_start = f32[8] all-reduce-start(input),
          replica_groups={{0,1},{2,3}}, to_apply=add,
          backend_config={"collective_backend_config":
              {"is_pipelined":true,"is_spmd_generated":true}}
      ROOT all_reduce_done = f32[8] all-reduce-done(all_reduce_start)
    }
  )"));

  const HloInstruction* all_reduce_start =
      module->entry_computation()->GetInstructionWithName("all_reduce_start");
  ASSERT_NE(all_reduce_start, nullptr);

  ModuleAnnotations annotations(*module, TraceAnnotationLevel::kDetailed);
  const InstructionAnnotation& annotation =
      annotations.instructions.at(all_reduce_start->name());

  auto xprof_name = annotation.xprof_name();

  EXPECT_TRUE(annotation.has_detailed_annotations());
  EXPECT_TRUE(annotation.is_collective_annotation());

  EXPECT_THAT(xprof_name, HasSubstr("hlo_op=all_reduce_start"));
  EXPECT_THAT(xprof_name,
              HasSubstr(absl::StrCat("unique_hlo_op_id=",
                                     all_reduce_start->unique_id())));
  EXPECT_THAT(xprof_name, HasSubstr("replica_groups={{0;1};{2;3}}"));
  EXPECT_THAT(xprof_name, HasSubstr("is_pipelined=1"));
  EXPECT_THAT(xprof_name, HasSubstr("is_spmd_generated=1"));
}

TEST(AnnotationTest, ModuleAnnotationsWithStackFrames) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(R"(
    HloModule test

    FileNames
    1 "model.py"

    FunctionNames
    1 "forward"
    2 "layer"

    FileLocations
    1 {file_name_id=1 function_name_id=1 line=10 end_line=12 column=4 end_column=20}
    2 {file_name_id=1 function_name_id=2 line=42 end_line=42 column=8 end_column=30}

    StackFrames
    1 {file_location_id=1 parent_frame_id=1}
    2 {file_location_id=2 parent_frame_id=2}

    fused_computation {
      p0 = f32[4] parameter(0)
      p1 = f32[4] parameter(1)
      ROOT add = f32[4] add(p0, p1),
          metadata={op_type="add", op_name="jit(forward)/layer/add", stack_frame_id=2}
    }

    ENTRY main {
      a = f32[4] parameter(0)
      b = f32[4] parameter(1)
      ROOT fusion = f32[4] fusion(a, b), kind=kLoop, calls=fused_computation,
          metadata={op_type="fusion", op_name="jit(forward)/layer/fusion", stack_frame_id=2}
    }
  )"));

  ModuleAnnotations basic_annotations(*module, TraceAnnotationLevel::kBasic);
  EXPECT_EQ(basic_annotations.top_level.common_stack_frames(),
            tsl::profiler::DefaultProfilerDomain() == nullptr ? 0 : 1);
  EXPECT_EQ(basic_annotations.instructions.size(), module->instruction_count());
  const InstructionAnnotation& basic_fusion =
      basic_annotations.instructions.at("fusion");
  EXPECT_FALSE(basic_fusion.has_detailed_annotations());
  EXPECT_EQ(basic_fusion.nvtx_name(), basic_fusion.xprof_name());
  EXPECT_THAT(basic_fusion.xprof_name(), HasSubstr("hlo_op=fusion"));

  ModuleAnnotations detailed_annotations(*module,
                                         TraceAnnotationLevel::kDetailed);
  EXPECT_EQ(detailed_annotations.instructions.size(),
            module->instruction_count());
  const InstructionAnnotation& detailed_fusion =
      detailed_annotations.instructions.at("fusion");
  EXPECT_TRUE(detailed_fusion.has_detailed_annotations());
  EXPECT_FALSE(detailed_fusion.is_collective_annotation());
  EXPECT_EQ(detailed_fusion.nvtx_name(), basic_fusion.nvtx_name());
  EXPECT_THAT(detailed_fusion.xprof_name(), HasSubstr("op_type=fusion"));
  EXPECT_THAT(detailed_fusion.xprof_name(),
              HasSubstr("op_name=jit(forward)/layer/fusion"));
  EXPECT_THAT(detailed_fusion.xprof_name(), HasSubstr("source_file=model.py"));
  EXPECT_THAT(detailed_fusion.xprof_name(), HasSubstr("source_line=42"));
  EXPECT_THAT(detailed_fusion.xprof_name(), HasSubstr("shape=f32[4]"));
}

}  // namespace
}  // namespace xla::gpu
