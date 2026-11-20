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

#include "xla/service/gpu/execution_stream_assignment.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/side_effect_util.h"

namespace xla::gpu {
namespace {

class ExecutionStreamAssignmentTest : public HloHardwareIndependentTestBase {};

std::string AnnotatedCallStartModule(absl::string_view annotation,
                                     bool communication_scope = false) {
  return absl::StrCat(R"(
    HloModule m

    leaf {
      p0 = f32[2] parameter(0)
      ROOT add = f32[2] add(p0, p0)
    }

    ENTRY entry {
      p0 = f32[2] parameter(0)
      start = ((f32[2]), f32[2]) call-start(p0), to_apply=leaf,
        frontend_attributes={_xla_stream_annotation=")",
                      annotation, "\"",
                      communication_scope ? R"(, _collectives_group="")" : "",
                      R"(}
      ROOT done = f32[2] call-done(start)
    }
  )");
}

struct LocalityDomainAnnotationTestCase {
  const char* name;
  const char* annotation;
  uint64_t expected_domain_id;
};

class ValidLocalityDomainAnnotationTest
    : public ExecutionStreamAssignmentTest,
      public ::testing::WithParamInterface<LocalityDomainAnnotationTestCase> {};

TEST_P(ValidLocalityDomainAnnotationTest, AssignsLocalityDomainStream) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(
                           AnnotatedCallStartModule(GetParam().annotation)));

  ExecutionStreamAssignment assignment(module.get());
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(
          LocalityDomainStreamId(GetParam().expected_domain_id))));
}

INSTANTIATE_TEST_SUITE_P(
    LocalityDomainIds, ValidLocalityDomainAnnotationTest,
    ::testing::ValuesIn<LocalityDomainAnnotationTestCase>({
        {"Zero", "locality_domain:0", 0},
        {"LeadingZero", "locality_domain:01", 1},
    }),
    [](const ::testing::TestParamInfo<LocalityDomainAnnotationTestCase>& info) {
      return info.param.name;
    });

struct MalformedLocalityDomainAnnotationTestCase {
  const char* name;
  const char* annotation;
  const char* message_regex;
};

class MalformedLocalityDomainAnnotationTest
    : public ExecutionStreamAssignmentTest,
      public ::testing::WithParamInterface<
          MalformedLocalityDomainAnnotationTestCase> {};

TEST_P(MalformedLocalityDomainAnnotationTest, DiesWithCheckMessage) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(
                           AnnotatedCallStartModule(GetParam().annotation)));

  EXPECT_DEATH(ExecutionStreamAssignment(module.get()),
               GetParam().message_regex);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidSyntax, MalformedLocalityDomainAnnotationTest,
    ::testing::ValuesIn<MalformedLocalityDomainAnnotationTestCase>({
        {"MissingColon", "locality_domain",
         "Invalid locality-domain stream annotation: expected"},
        {"EmptySuffix", "locality_domain:",
         "Invalid locality-domain stream annotation: expected"},
        {"NegativeSign", "locality_domain:-1",
         "domain id must be an unsigned decimal integer"},
        {"PositiveSign", "locality_domain:+1",
         "domain id must be an unsigned decimal integer"},
        {"LeadingWhitespace", "locality_domain: 1",
         "domain id must be an unsigned decimal integer"},
        {"TrailingWhitespace", "locality_domain:1 ",
         "domain id must be an unsigned decimal integer"},
        {"Alphabetic", "locality_domain:abc",
         "domain id must be an unsigned decimal integer"},
        {"Decimal", "locality_domain:1.0",
         "domain id must be an unsigned decimal integer"},
        {"TrailingJunk", "locality_domain:1x",
         "domain id must be an unsigned decimal integer"},
        {"Uint64Overflow", "locality_domain:18446744073709551616",
         "domain id is out of range"},
    }),
    [](const ::testing::TestParamInfo<
        MalformedLocalityDomainAnnotationTestCase>& info) {
      return info.param.name;
    });

TEST_F(ExecutionStreamAssignmentTest,
       LocalityDomainAnnotationRejectsCommunicationScope) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(AnnotatedCallStartModule(
                           "locality_domain:0",
                           /*communication_scope=*/true)));

  EXPECT_DEATH(ExecutionStreamAssignment(module.get()),
               "cannot be assigned to a communication execution scope");
}

TEST_F(ExecutionStreamAssignmentTest, AsyncFusion) {
  const char* kModuleStr = R"(
    HloModule m

    // Leaf computations.
    leaf1 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
    leaf2 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
    leaf3 {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }

    // Entry computation that calls each of the leaves asynchronously.
    ENTRY entry {
      p0 = f32[2,2] parameter(0)
      start1 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf1
      start2 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf2
      start3 = ((f32[2,2]), f32[2,2], s32[]) fusion-start(p0),
          kind=kLoop, calls=leaf3
      update1 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start1)
      update2 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start2)
      update3 = ((f32[2,2]), f32[2,2], s32[]) fusion-update(start3)
      done1 = f32[2,2] fusion-done(update1)
      done2 = f32[2,2] fusion-done(update2)
      done3 = f32[2,2] fusion-done(update3)
      ROOT done = f32[2,2] custom-call(done1, done2, done3),
          custom_call_target="target"
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(),
      ExecutionStreamAssignment::Options{/*number_of_execution_streams=*/2});

  // Only start operations get execution stream IDs. With 2 compute streams,
  // start1 gets ComputationStreamId(0), start2 gets ComputationStreamId(1),
  // start3 wraps around to ComputationStreamId(0).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start2")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start3")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));

  // Update and done operations do not get assigned stream IDs (they inherit
  // from the parent scope at run time via structured concurrency).
  for (absl::string_view instruction :
       {"update1", "update2", "update3", "done1", "done2", "done3"}) {
    EXPECT_THAT(assignment.GetExecutionStreamId(
                    FindInstruction(module.get(), instruction)),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, CopyStartStreamIdTest) {
  const char* const hlo_copy_start_string = R"(
  HloModule Module

  ENTRY CopyStartAndCopyDone {
    p0 = f32[2,3]{1,0:S(7)} parameter(0)
    copy-start = (f32[2,3]{1,0:S(2)}, f32[2,3]{1,0:S(7)}, u32[]) copy-start(p0)
    ROOT copy-done = f32[2,3]{1,0:S(2)} copy-done(copy-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_copy_start_string));

  ExecutionStreamAssignment assignment(module.get());

  // D2D copy-start (neither endpoint is host memory) runs on a compute stream.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "copy-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
}

TEST_F(ExecutionStreamAssignmentTest, CopyStartD2HStreamId) {
  // Destination is in host memory (S(5)) → device-to-host → kMemcpyD2HStreamId.
  const char* const hlo = R"(
  HloModule Module

  ENTRY main {
    p0 = f32[1024]{0} parameter(0)
    copy-start = (f32[1024]{0:S(5)}, f32[1024]{0}, u32[]) copy-start(p0)
    ROOT copy-done = f32[1024]{0:S(5)} copy-done(copy-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  ExecutionStreamAssignment assignment(module.get());

  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "copy-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(kMemcpyD2HStreamId)));
}

TEST_F(ExecutionStreamAssignmentTest, CopyStartH2DStreamId) {
  // Source is in host memory (S(5)) → host-to-device → kMemcpyH2DStreamId.
  const char* const hlo = R"(
  HloModule Module

  ENTRY main {
    p0 = f32[1024]{0:S(5)} parameter(0)
    copy-start = (f32[1024]{0}, f32[1024]{0:S(5)}, u32[]) copy-start(p0)
    ROOT copy-done = f32[1024]{0} copy-done(copy-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  ExecutionStreamAssignment assignment(module.get());

  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "copy-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(kMemcpyH2DStreamId)));
}

TEST_F(ExecutionStreamAssignmentTest, AsyncDusToHostMemory) {
  // async-start wrapping a DUS whose output is in host memory → D2H.
  const char* const hlo = R"(
  HloModule Module

  dus_computation {
    base = f32[4]{0:S(5)} parameter(0)
    update = f32[1]{0} parameter(1)
    idx = s32[] parameter(2)
    ROOT result = f32[4]{0:S(5)} dynamic-update-slice(base, update, idx)
  }

  ENTRY main {
    p0 = f32[4]{0:S(5)} parameter(0)
    p1 = f32[1]{0} parameter(1)
    p2 = s32[] parameter(2)
    start = ((f32[4]{0:S(5)}, f32[1]{0}, s32[]), f32[4]{0:S(5)}, u32[]) async-start(p0, p1, p2), calls=dus_computation
    ROOT done = f32[4]{0:S(5)} async-done(start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  ExecutionStreamAssignment assignment(module.get());

  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(kMemcpyD2HStreamId)));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, AsyncFusedDusToHostMemory) {
  // async-start wrapping a kLoop fusion whose output is in host memory (the
  // post-StreamAttributeAnnotator form of a D2H host-memory DUS).
  const char* const hlo = R"(
  HloModule Module

  wrapped_dus_computation {
    base = f32[4]{0:S(5)} parameter(0)
    update = f32[1]{0} parameter(1)
    idx = s32[] parameter(2)
    ROOT dus = f32[4]{0:S(5)} dynamic-update-slice(base, update, idx)
  }

  fused_dus_async {
    p0 = f32[4]{0:S(5)} parameter(0)
    p1 = f32[1]{0} parameter(1)
    p2 = s32[] parameter(2)
    ROOT fused = f32[4]{0:S(5)} fusion(p0, p1, p2), kind=kLoop,
        calls=wrapped_dus_computation
  }

  ENTRY main {
    p0 = f32[4]{0:S(5)} parameter(0)
    p1 = f32[1]{0} parameter(1)
    p2 = s32[] parameter(2)
    start = ((f32[4]{0:S(5)}, f32[1]{0}, s32[]), f32[4]{0:S(5)}, u32[]) async-start(p0, p1, p2), calls=fused_dus_async
    ROOT done = f32[4]{0:S(5)} async-done(start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  ExecutionStreamAssignment assignment(module.get());

  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(kMemcpyD2HStreamId)));
}

TEST_F(ExecutionStreamAssignmentTest, AsyncDsFromHostMemory) {
  // async-start wrapping a DS whose first operand is in host memory → H2D.
  const char* const hlo = R"(
  HloModule Module

  ds_computation {
    src = f32[4]{0:S(5)} parameter(0)
    idx = s32[] parameter(1)
    ROOT result = f32[1]{0} dynamic-slice(src, idx), dynamic_slice_sizes={1}
  }

  ENTRY main {
    p0 = f32[4]{0:S(5)} parameter(0)
    p1 = s32[] parameter(1)
    start = ((f32[4]{0:S(5)}, s32[]), f32[1]{0}, u32[]) async-start(p0, p1), calls=ds_computation
    ROOT done = f32[1]{0} async-done(start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  ExecutionStreamAssignment assignment(module.get());

  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(kMemcpyH2DStreamId)));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, FusionComputations) {
  const char* kModuleStr = R"(
    HloModule m

    reduce {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      p0 = f32[4] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[] reduce(p0, c0), dimensions={0}, to_apply=reduce
    }

    // Entry computation with a synchronous fusion.
    ENTRY entry {
      p0 = f32[4] parameter(0)
      ROOT done = f32[] fusion(p0), kind=kLoop, calls=fusion
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // Synchronous instructions (including fusions) do not get execution stream
  // IDs. They run on the default stream or inherit from a parent scope.
  for (const HloInstruction* instruction :
       FindComputation(module.get(), "entry")->instructions()) {
    EXPECT_THAT(assignment.GetExecutionStreamId(instruction),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, UnreachableComputation) {
  const char* kModuleStr = R"(
    HloModule m

    // Unreachable computation.
    unreachable {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }

    ENTRY entry {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(module.get());

  // No scope-start operations exist, so nothing gets assigned.
  for (const HloInstruction* instruction :
       FindComputation(module.get(), "entry")->instructions()) {
    EXPECT_THAT(assignment.GetExecutionStreamId(instruction),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest, ExplicitStreams) {
  const char* kModuleStr = R"(
  HloModule m
  %gemm1 (x: f32[2048,2048], y: f32[2048,2048]) -> f32[2048,2048] {
    %y = f32[2048,2048]{1,0} parameter(1)
    %x = f32[2048,2048]{1,0} parameter(0)
    %custom-call.1 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false}}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.1), index=0
  }

  %gemm2 (x: f32[2048,2048], y: f32[2048,2048]) -> f32[2048,2048] {
    %y = f32[2048,2048]{1,0} parameter(1)
    %x = f32[2048,2048]{1,0} parameter(0)
    %custom-call.2 = (f32[2048,2048]{1,0}, s8[33554432]{0}) custom-call(f32[2048,2048]{1,0} %x, f32[2048,2048]{1,0} %y), custom_call_target="__cublas$gemm", backend_config={"gemm_backend_config":{"alpha_real":1,"alpha_imag":0,"beta":0,"dot_dimension_numbers":{"lhs_contracting_dimensions":["1"],"rhs_contracting_dimensions":["0"],"lhs_batch_dimensions":[],"rhs_batch_dimensions":[]},"precision_config":{"operand_precision":["DEFAULT","DEFAULT"],"algorithm":"ALG_UNSET"},"epilogue":"DEFAULT","damax_output":false,"lhs_stride":"4194304","rhs_stride":"4194304","grad_x":false,"grad_y":false}}
    ROOT %get-tuple-element = f32[2048,2048]{1,0} get-tuple-element((f32[2048,2048]{1,0}, s8[33554432]{0}) %custom-call.2), index=0
  }

  ENTRY %entry (y: f32[2048,2048], x: f32[2048,2048]) -> f32[2048,2048] {
    %x = f32[2048,2048]{1,0} parameter(1), metadata={op_name="b" scheduling_name="x"}
    %y = f32[2048,2048]{1,0} parameter(0), metadata={op_name="a" scheduling_name="y"}
    %call-start = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %y, f32[2048,2048]{1,0} %x), to_apply=%gemm1, frontend_attributes={_xla_stream_annotation="1"}
    %call-done = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start)
    %call-start.2 = ((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) call-start(f32[2048,2048]{1,0} %y, f32[2048,2048]{1,0} %call-done), to_apply=%gemm2, frontend_attributes={_xla_stream_annotation="2"}
    ROOT %call-done-2 = f32[2048,2048]{1,0} call-done(((f32[2048,2048]{1,0}, f32[2048,2048]{1,0}), f32[2048,2048]{1,0}) %call-start.2)
}
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(),
      ExecutionStreamAssignment::Options{/*number_of_execution_streams=*/4});

  // call-start has explicit annotation "1" -> ComputationStreamId(1).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));

  // call-start.2 has explicit annotation "2" -> ComputationStreamId(2).
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start.2")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(2))));

  // Done operations don't get stream IDs.
  EXPECT_THAT(assignment.GetExecutionStreamId(
                  FindInstruction(module.get(), "call-done")),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest,
       LocalityDomainDoesNotConsumeRoundRobinStream) {
  const char* kModuleStr = R"(
    HloModule m

    leaf0 {
      p0 = f32[2] parameter(0)
      ROOT add = f32[2] add(p0, p0)
    }
    leaf1 {
      p0 = f32[2] parameter(0)
      ROOT add = f32[2] add(p0, p0)
    }
    leaf2 {
      p0 = f32[2] parameter(0)
      ROOT add = f32[2] add(p0, p0)
    }

    ENTRY entry {
      p0 = f32[2] parameter(0)
      start0 = ((f32[2]), f32[2]) call-start(p0), to_apply=leaf0
      done0 = f32[2] call-done(start0)
      locality = ((f32[2]), f32[2]) call-start(done0), to_apply=leaf1,
        frontend_attributes={_xla_stream_annotation="locality_domain:7"}
      locality_done = f32[2] call-done(locality)
      start1 = ((f32[2]), f32[2]) call-start(locality_done), to_apply=leaf2
      ROOT done1 = f32[2] call-done(start1)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/2,
                     /*number_of_communication_execution_streams=*/1});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start0")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "locality")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(LocalityDomainStreamId(7))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "start1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
}

TEST_F(ExecutionStreamAssignmentTest, AsyncCollectiveTest) {
  const char* const hlo_string = R"(
  HloModule m, is_scheduled=true
    reduce {
      x = f32[] parameter(0)
      y = f32[] parameter(1)
      ROOT _ = f32[] add(x, y)
    }
    ENTRY main {
      p0 = f32[] parameter(0)
      p1 = f32[2] parameter(1)
      p2 = f32[2] parameter(2)
      ar-start = f32[] all-reduce-start(p0), to_apply=reduce
      rs-start = ((f32[2]), f32[1]) reduce-scatter-start(p1), to_apply=reduce, dimensions={0}
      add.0 = f32[2] add(p1, p2)
      ar-done = f32[] all-reduce-done(ar-start)
      rs-done = f32[1] reduce-scatter-done(rs-start)
      ROOT _ = (f32[], f32[1], f32[2]) tuple(ar-done, rs-done, add.0)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  // With 4 compute streams and 2 collective streams, ar-start and rs-start
  // get CommunicationStreamId(0) and CommunicationStreamId(1).
  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/2});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ar-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));

  // Redo with 1 collective stream: both should get CommunicationStreamId(0).
  assignment = ExecutionStreamAssignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/1});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ar-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));

  // Done operations don't get stream IDs.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "ar-done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, CollectiveDomainRoundRobin) {
  const char* const hlo_string = R"(
  HloModule m

  reduce {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT sum = f32[] add(x, y)
  }

  ENTRY main {
    p0 = f32[] parameter(0)
    p1 = f32[2] parameter(1)
    ar-start = f32[] all-reduce-start(p0), to_apply=reduce
    rs-start = ((f32[2]), f32[1]) reduce-scatter-start(p1),
      to_apply=reduce, dimensions={0}
    rs1-start = ((f32[2]), f32[1]) reduce-scatter-start(p1),
      to_apply=reduce, dimensions={0}
    rs2-start = ((f32[2]), f32[1]) reduce-scatter-start(p1),
      to_apply=reduce, dimensions={0}
    rs3-start = ((f32[2]), f32[1]) reduce-scatter-start(p1),
      to_apply=reduce, dimensions={0}
    rs4-start = ((f32[2]), f32[1]) reduce-scatter-start(p1),
      to_apply=reduce, dimensions={0}
    ar-done = f32[] all-reduce-done(ar-start)
    rs-done = f32[1] reduce-scatter-done(rs-start)
    rs1-done = f32[1] reduce-scatter-done(rs1-start)
    rs2-done = f32[1] reduce-scatter-done(rs2-start)
    rs3-done = f32[1] reduce-scatter-done(rs3-start)
    rs4-done = f32[1] reduce-scatter-done(rs4-start)
    ROOT result = (f32[], f32[1], f32[1], f32[1], f32[1], f32[1])
      tuple(ar-done, rs-done, rs1-done, rs2-done, rs3-done, rs4-done)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  GpuBackendConfig scale_up_fabric_config;
  scale_up_fabric_config.mutable_collective_backend_config()
      ->set_communication_domain(kScaleUpFabricCollectiveDomain);
  for (absl::string_view name : {"ar-start", "rs1-start", "rs3-start"}) {
    HloInstruction* start = FindInstruction(module.get(), name);
    ASSERT_OK(start->set_backend_config(scale_up_fabric_config));
  }

  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/2});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ar-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(3))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  // Each domain advances independently and wraps within its own two-stream
  // pool.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs1-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(4))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs2-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs3-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(3))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "rs4-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
}

TEST_F(ExecutionStreamAssignmentTest,
       ExplicitCollectivesGroupUsesCommunicationStream) {
  const char* const hlo_string = R"(
  HloModule m

  comms {
    p0 = f32[1] parameter(0)
    ag = f32[1] all-gather(p0), dimensions={0}
    cp = f32[1] collective-permute(p0), source_target_pairs={{0,1}}
    ROOT result = (f32[1], f32[1]) tuple(ag, cp)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    group-start = ((f32[1]), (f32[1], f32[1])) async-start(p0),
      calls=comms, frontend_attributes={_collectives_group=""}
    ROOT group-done = (f32[1], f32[1]) async-done(group-start)
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ExecutionStreamAssignment assignment(module.get());
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "group-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));

  HloInstruction* group_start = FindInstruction(module.get(), "group-start");
  GpuBackendConfig config;
  config.mutable_collective_backend_config()->set_communication_domain(
      kScaleUpFabricCollectiveDomain);
  ASSERT_OK(group_start->set_backend_config(config));
  assignment = ExecutionStreamAssignment(module.get());
  EXPECT_THAT(
      assignment.GetExecutionStreamId(group_start),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(3))));

  group_start->set_frontend_attribute(kXlaStreamAnnotationAttr, "7");
  assignment = ExecutionStreamAssignment(module.get());
  EXPECT_THAT(
      assignment.GetExecutionStreamId(group_start),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(7))));
  EXPECT_THAT(assignment.GetExecutionStreamId(
                  FindInstruction(module.get(), "group-done")),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest,
       CollectiveStreamCallAssignsCommunicationStreamToWrappedCollective) {
  const char* const hlo_string = R"(
  HloModule m

  body {
    p0 = f32[1] parameter(0)
    ag-start = (f32[1], f32[1]) all-gather-start(p0), dimensions={0}
    ROOT ag-done = f32[1] all-gather-done(ag-start)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    call-start = ((f32[1]), f32[1]) call-start(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="collective"}
    ROOT call-done = f32[1] call-done(call-start),
      frontend_attributes={_xla_stream_annotation="collective"}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/2});

  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ag-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "ag-done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(assignment.GetExecutionStreamId(
                  FindInstruction(module.get(), "call-done")),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(
    ExecutionStreamAssignmentTest,
    CollectiveStreamCallSharesSingleCommunicationStreamWithWrappedCollective) {
  const char* const hlo_string = R"(
  HloModule m

  body {
    p0 = f32[1] parameter(0)
    ag-start = (f32[1], f32[1]) all-gather-start(p0), dimensions={0}
    ROOT ag-done = f32[1] all-gather-done(ag-start)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    call-start = ((f32[1]), f32[1]) call-start(p0), to_apply=body,
      frontend_attributes={_xla_stream_annotation="collective"}
    ROOT call-done = f32[1] call-done(call-start),
      frontend_attributes={_xla_stream_annotation="collective"}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/1});

  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "call-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ag-start")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "ag-done")),
      absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(assignment.GetExecutionStreamId(
                  FindInstruction(module.get(), "call-done")),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ExecutionStreamAssignmentTest, PipelinedSendRecv) {
  const char* kModuleStr = R"(
    HloModule test, num_partitions=2

    cond {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      c2 = u32[] constant(2)
      ROOT cmp = pred[] compare(i, c2), direction=LT
    }

    body {
      param = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          parameter(0)
      i = u32[] get-tuple-element(param), index=0
      send_ctx = get-tuple-element(param), index=1
      recv_ctx = get-tuple-element(param), index=2
      send_done_inner = token[] send-done(send_ctx), channel_id=1
      recv_done_inner = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      data = get-tuple-element(recv_done_inner), index=0
      after_all = token[] after-all()
      send_ctx_inner = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx_inner = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2
      c1 = u32[] constant(1)
      i_ = u32[] add(i, c1)
      ROOT result = (u32[], (f32[2,2], u32[], token[]),
          (f32[2,2], u32[], token[])) tuple(i_, send_ctx_inner, recv_ctx_inner)
    }

    ENTRY test_computation {
      data = f32[2,2] parameter(0)
      i = u32[] constant(0)
      after_all = token[] after-all()
      send_ctx_ = (f32[2,2], u32[], token[]) send(data, after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=1
      recv_ctx_ = (f32[2,2], u32[], token[]) recv(after_all),
          frontend_attributes={_xla_send_recv_source_target_pairs={{0,1}}},
          channel_id=2
      init = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          tuple(i, send_ctx_, recv_ctx_)
      while = (u32[], (f32[2,2], u32[], token[]), (f32[2,2], u32[], token[]))
          while(init), condition=cond, body=body
      send_ctx = get-tuple-element(while), index=1
      recv_ctx = get-tuple-element(while), index=2
      send_done = token[] send-done(send_ctx), channel_id=1
      recv_done = (f32[2,2], token[]) recv-done(recv_ctx), channel_id=2
      ROOT data_ = get-tuple-element(recv_done), index=0
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kModuleStr));

  ExecutionStreamAssignment assignment(
      module.get(), ExecutionStreamAssignment::Options{4, 4});

  // Only the canonical send/recv starts get stream IDs.
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "send_ctx_")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "recv_ctx_")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));

  // Non-canonical send/recv and done operations don't get stream IDs.
  for (absl::string_view instruction :
       {"send_ctx_inner", "recv_ctx_inner", "send_done_inner",
        "recv_done_inner", "send_done", "recv_done"}) {
    EXPECT_THAT(assignment.GetExecutionStreamId(
                    FindInstruction(module.get(), instruction)),
                absl_testing::StatusIs(absl::StatusCode::kNotFound));
  }
}

TEST_F(ExecutionStreamAssignmentTest,
       MultipleStreamAnnotatedCallsAssignStreamsToAllWrappedCollectives) {
  const char* const hlo_string = R"(
  HloModule m

  body1 {
    p0 = f32[1] parameter(0)
    ag-start.0 = (f32[1], f32[1]) all-gather-start(p0), dimensions={0}
    ROOT ag-done.0 = f32[1] all-gather-done(ag-start.0)
  }

  body2 {
    p0 = f32[1] parameter(0)
    ag-start.1 = (f32[1], f32[1]) all-gather-start(p0), dimensions={0}
    ROOT ag-done.1 = f32[1] all-gather-done(ag-start.1)
  }

  ENTRY main {
    p0 = f32[1] parameter(0)
    cs0 = ((f32[1]), f32[1]) call-start(p0),
      to_apply=body1,
      frontend_attributes={_xla_stream_annotation="collective"}
    cd0 = f32[1] call-done(cs0),
      frontend_attributes={_xla_stream_annotation="collective"}
    cs1 = ((f32[1]), f32[1]) call-start(cd0),
      to_apply=body2,
      frontend_attributes={_xla_stream_annotation="collective"}
    ROOT cd1 = f32[1] call-done(cs1),
      frontend_attributes={_xla_stream_annotation="collective"}
  }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/2});

  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "cs0")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "cs1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ag-start.0")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "ag-start.1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(CommunicationStreamId(1))));
}

TEST_F(ExecutionStreamAssignmentTest,
       MultipleD2DCopyStartsShareSingleDedicatedStream) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    ENTRY main {
      p0 = f32[16]{0} parameter(0)
      p1 = f32[16]{0} parameter(1)
      cs0 = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p0)
      cs1 = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p1)
      cs2 = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p0)
      cd0 = f32[16]{0} copy-done(cs0)
      cd1 = f32[16]{0} copy-done(cs1)
      cd2 = f32[16]{0} copy-done(cs2)
      ROOT out = (f32[16]{0}, f32[16]{0}, f32[16]{0}) tuple(cd0, cd1, cd2)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ExecutionStreamAssignment assignment(module.get());
  for (absl::string_view name : {"cs0", "cs1", "cs2"}) {
    EXPECT_THAT(
        assignment.GetExecutionStreamId(FindInstruction(module.get(), name)),
        absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  }

  // When dedicated memcpy streams are disabled, D2D copy-start falls back to
  // round-robin assignment across the compute stream pool.
  ExecutionStreamAssignment round_robin(
      module.get(), {/*number_of_compute_execution_streams=*/2,
                     /*number_of_communication_execution_streams=*/1,
                     /*enable_dedicated_memcpy_streams=*/false});
  EXPECT_THAT(
      round_robin.GetExecutionStreamId(FindInstruction(module.get(), "cs0")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  EXPECT_THAT(
      round_robin.GetExecutionStreamId(FindInstruction(module.get(), "cs1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
  EXPECT_THAT(
      round_robin.GetExecutionStreamId(FindInstruction(module.get(), "cs2")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
}

TEST_F(ExecutionStreamAssignmentTest,
       ExplicitlyAnnotatedD2DCopyAvoidsUnannotatedD2DStreamCollision) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    ENTRY main {
      p0 = f32[16]{0} parameter(0)
      p1 = f32[16]{0} parameter(1)
      cs_annotated = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p0),
          frontend_attributes={_xla_stream_annotation="0"}
      cs0 = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p1)
      cs1 = (f32[16]{0}, f32[16]{0}, u32[]) copy-start(p0)
      cd_annotated = f32[16]{0} copy-done(cs_annotated)
      cd0 = f32[16]{0} copy-done(cs0)
      cd1 = f32[16]{0} copy-done(cs1)
      ROOT out = (f32[16]{0}, f32[16]{0}, f32[16]{0}) tuple(
          cd_annotated, cd0, cd1)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  ExecutionStreamAssignment assignment(module.get());
  EXPECT_THAT(
      assignment.GetExecutionStreamId(
          FindInstruction(module.get(), "cs_annotated")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  for (absl::string_view name : {"cs0", "cs1"}) {
    EXPECT_THAT(
        assignment.GetExecutionStreamId(FindInstruction(module.get(), name)),
        absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
  }
}

TEST_F(ExecutionStreamAssignmentTest,
       DynamicSliceCopyFusionsShareSingleStreamDisjointFromCompute) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule m

    compute_leaf0 {
      p0 = f32[4] parameter(0)
      ROOT add = f32[4] add(p0, p0)
    }

    compute_leaf1 {
      p0 = f32[4] parameter(0)
      ROOT mul = f32[4] multiply(p0, p0)
    }

    dus_copy0 {
      base = f32[8] parameter(0)
      upd = f32[4] parameter(1)
      c0 = s32[] constant(0)
      ROOT dus = f32[8] dynamic-update-slice(base, upd, c0),
          backend_config={"dynamic_slice_config":
              {"linear":{"byte_offset":"0","byte_stride":"0"}}}
    }

    ds_copy1 {
      src = f32[8] parameter(0)
      c1 = s32[] constant(4)
      ROOT ds = f32[4] dynamic-slice(src, c1), dynamic_slice_sizes={4},
          backend_config={"dynamic_slice_config":
              {"linear":{"byte_offset":"16","byte_stride":"0"}}}
    }

    ENTRY main {
      p_base = f32[8] parameter(0)
      p_vec = f32[4] parameter(1)
      comp0 = ((f32[4]), f32[4], u32[]) fusion-start(p_vec),
          kind=kLoop, calls=compute_leaf0
      dus0 = ((f32[8], f32[4]), f32[8], u32[]) fusion-start(p_base, p_vec),
          kind=kLoop, calls=dus_copy0
      comp1 = ((f32[4]), f32[4], u32[]) fusion-start(p_vec),
          kind=kLoop, calls=compute_leaf1
      ds1 = ((f32[8]), f32[4], u32[]) fusion-start(p_base),
          kind=kLoop, calls=ds_copy1
      cs_d2d = (f32[4], f32[4], u32[]) copy-start(p_vec)
      comp0_done = f32[4] fusion-done(comp0)
      dus0_done = f32[8] fusion-done(dus0)
      comp1_done = f32[4] fusion-done(comp1)
      ds1_done = f32[4] fusion-done(ds1)
      cs_d2d_done = f32[4] copy-done(cs_d2d)
      ROOT out = (f32[4], f32[8], f32[4], f32[4], f32[4]) tuple(
          comp0_done, dus0_done, comp1_done, ds1_done, cs_d2d_done)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));

  // Compute fusions use streams 0 and 1 without being shifted by interleaved
  // D2D copies; all D2D copy fusions and copy-starts share stream 2.
  ExecutionStreamAssignment assignment(
      module.get(), {/*number_of_compute_execution_streams=*/4,
                     /*number_of_communication_execution_streams=*/1});
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "comp0")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(0))));
  EXPECT_THAT(
      assignment.GetExecutionStreamId(FindInstruction(module.get(), "comp1")),
      absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(1))));
  for (absl::string_view name : {"dus0", "ds1", "cs_d2d"}) {
    EXPECT_THAT(
        assignment.GetExecutionStreamId(FindInstruction(module.get(), name)),
        absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(2))));
  }

  // When all compute pool streams (0..1) are occupied, the shared D2D stream
  // still gets a disjoint stream ID (2).
  ExecutionStreamAssignment full_pool(
      module.get(), {/*number_of_compute_execution_streams=*/2,
                     /*number_of_communication_execution_streams=*/1});
  for (absl::string_view name : {"dus0", "ds1", "cs_d2d"}) {
    EXPECT_THAT(
        full_pool.GetExecutionStreamId(FindInstruction(module.get(), name)),
        absl_testing::IsOkAndHolds(ExecutionStreamId(ComputationStreamId(2))));
  }
}

}  // namespace
}  // namespace xla::gpu
