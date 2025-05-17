/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/model/matmul_interpolator.h"

#include <time.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::Test;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::ValuesIn;

struct DotContext {
  HloInstruction* dot;
  std::unique_ptr<HloModule> module;
};

struct DotSpec {
  int b;
  int m;
  int n;
  int k;
  int64_t clock_cycles;
};

struct ParametrizedTestCase {
  std::string test_name;
  DotSpec spec;
  absl::Duration expected_duration;
};

class MatmulInterpolatorParamTest : public TestWithParam<ParametrizedTestCase> {
 public:
  MatmulInterpolatorParamTest()
      : device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo()) {}

  void SetUp() override {
    absl::StatusOr<HloInstructionProfileList> profiles =
        DotInterpolationSpace(interpolation_space_);
    CHECK_OK(profiles.status()) << "Cannot generate interpolation space.";
    absl::StatusOr<std::unique_ptr<const MatmulInterpolator>> interpolator =
        MatmulInterpolator::Create(*std::move(profiles), device_info_);
    CHECK_OK(interpolator.status()) << "Cannot construct interpolator.";
    interpolator_ = std::move(*interpolator);
  }

 protected:
  absl::StatusOr<DotContext> Dot(int b, int m, int n, int k) {
    absl::string_view kTemplate = R"(
    HloModule m

    ENTRY r {
      lhs = f32[$0,$1,$2] parameter(0)
      rhs = f32[$0,$2,$3] parameter(1)
      ROOT _ = f32[$0,$1,$3] dot(lhs,rhs),
       lhs_contracting_dims={2}, rhs_contracting_dims={1},
       lhs_batch_dims={0}, rhs_batch_dims={0}
    })";
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnUnverifiedModule(
                            absl::Substitute(kTemplate, b, m, k, n)));
    return DotContext{
        /*dot=*/module->entry_computation()->root_instruction(),
        /*module=*/std::move(module),
    };
  }

  void AddProfileEntry(DotContext dot_context, int64_t clock_cycles,
                       HloInstructionProfileList& list) {
    HloInstructionProfile profile;
    *profile.mutable_instruction() = dot_context.dot->ToProto();
    *profile.add_operands() = dot_context.dot->operands()[0]->ToProto();
    *profile.add_operands() = dot_context.dot->operands()[1]->ToProto();
    profile.set_clock_cycles(clock_cycles);
    *list.add_entries() = std::move(profile);
  }

  absl::StatusOr<HloInstructionProfileList> DotInterpolationSpace(
      absl::Span<const DotSpec> specs) {
    HloInstructionProfileList list;
    for (DotSpec spec : specs) {
      TF_ASSIGN_OR_RETURN(DotContext dot_context,
                          Dot(spec.b, spec.m, spec.n, spec.k));
      AddProfileEntry(std::move(dot_context), spec.clock_cycles, list);
    }
    return list;
  }

  const MatmulInterpolator& interpolator() { return *interpolator_; }

 private:
  int64_t ClockCycles(absl::Duration runtime) {
    return absl::ToInt64Nanoseconds(runtime) * device_info_.clock_rate_ghz();
  }

  const se::DeviceDescription device_info_;
  const std::vector<DotSpec> interpolation_space_ = {
      DotSpec{
          /*b=*/1,
          /*m=*/256,
          /*n=*/1024,
          /*k=*/512,
          /*clock_cycles=*/ClockCycles(absl::Seconds(1)),
      },
      DotSpec{
          /*b=*/1,
          /*m=*/256,
          /*n=*/2048,
          /*k=*/512,
          /*clock_cycles=*/ClockCycles(absl::Seconds(2)),
      },
      DotSpec{
          /*b=*/1,
          /*m=*/64,
          /*n=*/2048,
          /*k=*/512,
          /*clock_cycles=*/ClockCycles(absl::Seconds(3)),
      },
      DotSpec{
          /*b=*/2,
          /*m=*/256,
          /*n=*/1024,
          /*k=*/512,
          /*clock_cycles=*/ClockCycles(absl::Seconds(4)),
      },
      DotSpec{
          /*b=*/2,
          /*m=*/256,
          /*n=*/2048,
          /*k=*/512,
          /*clock_cycles=*/ClockCycles(absl::Seconds(5)),
      },
  };
  std::unique_ptr<const MatmulInterpolator> interpolator_;
};

TEST_P(MatmulInterpolatorParamTest,
       MatmulInteprolatorNextNeighbourInterpolation) {
  const auto& [_, spec, expected_duration] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(DotContext context,
                          Dot(spec.b, spec.m, spec.n, spec.k));
  EXPECT_EQ(absl::Trunc(*interpolator().EstimatedRuntime(*context.dot),
                        absl::Milliseconds(1)),
            expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterpolatorTestInstantiation, MatmulInterpolatorParamTest,
    ValuesIn<ParametrizedTestCase>({
        /*Interpolates to b=1,m=256,n=1024,k=512*/
        {
            /*test_name=*/"smallest_dims_extrapolate", /*spec=*/
            {
                /*b=*/1,
                /*m=*/64,
                /*n=*/64,
                /*k=*/64,
            },
            /*expected_duration=*/absl::Milliseconds(1),
        },
        /*Interpolates to b=2,m=256,n=2048,k=512*/
        {
            /*test_name=*/"highest_dims_extrapolate", /*spec=*/
            {
                /*b=*/4,
                /*m=*/512,
                /*n=*/2048,
                /*k=*/1024,
            },
            /*expected_duration=*/absl::Seconds(40),
        },
        /*Interpolates to b=1,m=64,n=2048,k=512*/
        {
            /*test_name=*/"m_interpolate", /*spec=*/
            {
                /*b=*/2,
                /*m=*/128,
                /*n=*/2048,
                /*k=*/512,
            },
            /*expected_duration=*/absl::Seconds(12),
        },
        /*Interpolates to b=2,m=256,n=2048,k=512*/
        {
            /*test_name=*/"m_extrapolate", /*spec=*/
            {
                /*b=*/2,
                /*m=*/512,
                /*n=*/2048,
                /*k=*/512,
            },
            /*expected_duration=*/absl::Seconds(10),
        },
        /*Interpolates to b=2,m=256,n=2048,k=512*/
        {
            /*test_name=*/"n_extrapolate", /*spec=*/
            {
                /*b=*/2,
                /*m=*/256,
                /*n=*/4096,
                /*k=*/512,
            },
            /*expected_duration=*/absl::Seconds(10),
        },
        /*Interpolates to b=2,m=256,n=2048,k=512*/
        {
            /*test_name=*/"k_extrapolate", /*spec=*/
            {
                /*b=*/2,
                /*m=*/256,
                /*n=*/2048,
                /*k=*/1024,
            },
            /*expected_duration=*/absl::Seconds(10),
        },
    }),
    [](const TestParamInfo<MatmulInterpolatorParamTest::ParamType>& info) {
      return info.param.test_name;
    });

class MatmulInterpolatorDefaultTableTest
    : public TestWithParam<ParametrizedTestCase> {
 public:
  MatmulInterpolatorDefaultTableTest()
      : device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo(
            se::CudaComputeCapability(9, 0))) {}

  void SetUp() override {
    absl::StatusOr<std::unique_ptr<MatmulInterpolator>> interpolator_status =
        MatmulInterpolator::Create(device_info_);
    CHECK_OK(interpolator_status.status())
        << "Cannot construct interpolator from default table.";
    interpolator_ = std::move(*interpolator_status);
  }

 protected:
  // Generates a Dot HLO instruction with BF16 data type.
  absl::StatusOr<DotContext> DotBF16(int b, int m, int n, int k) {
    // Template uses $0=b, $1=m, $2=k, $3=n for dimensions.
    absl::string_view kTemplate = R"(
    HloModule m

    ENTRY r {
      lhs = bf16[$0,$1,$2] parameter(0)
      rhs = bf16[$0,$2,$3] parameter(1)
      ROOT _ = bf16[$0,$1,$3] dot(lhs,rhs),
       lhs_contracting_dims={2}, rhs_contracting_dims={1},
       lhs_batch_dims={0}, rhs_batch_dims={0}
    })";
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnUnverifiedModule(
                            absl::Substitute(kTemplate, b, m, k, n)));
    return DotContext{
        /*dot=*/module->entry_computation()->root_instruction(),
        /*module=*/std::move(module),
    };
  }

  // Generates a Dot HLO instruction with FP8 data type.
  absl::StatusOr<DotContext> DotFP8(int b, int m, int n, int k) {
    // Template uses $0=b, $1=m, $2=k, $3=n for dimensions.
    absl::string_view kTemplate = R"(
    HloModule m

    ENTRY r {
      lhs = s8[$0,$1,$2] parameter(0)
      rhs = s8[$0,$2,$3] parameter(1)
      ROOT _ = s8[$0,$1,$3] dot(lhs,rhs),
       lhs_contracting_dims={2}, rhs_contracting_dims={1},
       lhs_batch_dims={0}, rhs_batch_dims={0}
    })";
    TF_ASSIGN_OR_RETURN(auto module,
                        ParseAndReturnUnverifiedModule(
                            absl::Substitute(kTemplate, b, m, k, n)));
    return DotContext{
        /*dot=*/module->entry_computation()->root_instruction(),
        /*module=*/std::move(module),
    };
  }

  const MatmulInterpolator& interpolator() { return *interpolator_; }

 private:
  const se::DeviceDescription device_info_;
  std::unique_ptr<const MatmulInterpolator> interpolator_;
};

using BF16Test = MatmulInterpolatorDefaultTableTest;

TEST_P(BF16Test, EstimatesRuntimeForBF16) {
  const auto& [_, spec, expected_duration] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(DotContext context,
                          DotBF16(spec.b, spec.m, spec.n, spec.k));
  // Compare with nanosecond precision.
  EXPECT_EQ(absl::Trunc(*interpolator().EstimatedRuntime(*context.dot),
                        absl::Microseconds(1)),
            expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterpolatorDefaultTableTestInstantiation, BF16Test,
    ValuesIn<ParametrizedTestCase>({
        {
            /*test_name=*/"exact_match1_bf16",
            /*spec=*/
            {/*b=*/1, /*m=*/1024, /*n=*/4096, /*k=*/512, /*clock_cycles=*/0},
            /*expected_duration=*/absl::Microseconds(12),
        },
        {
            /*test_name=*/"exact_match2_bf16",
            /*spec=*/
            {/*b=*/4, /*m=*/256, /*n=*/1024, /*k=*/256, /*clock_cycles=*/0},
            /*expected_duration=*/absl::Microseconds(6),
        },
        {
            /*test_name=*/"exact_match3_bf16",
            /*spec=*/
            {/*b=*/1, /*m=*/4096, /*n=*/2048, /*k=*/4096, /*clock_cycles=*/0},
            /*expected_duration=*/absl::Microseconds(89),
        },
        {
            /*test_name=*/"extrapolate_small_bf16",
            /*spec=*/
            {/*b=*/1, /*m=*/64, /*n=*/64, /*k=*/64, /*clock_cycles=*/0},
            // Expected duration based on nearest point (1,256,256,256)
            // flops/sec and scaling by new dimensions.
            /*expected_duration=*/absl::Microseconds(0),
        },
        {
            /*test_name=*/"extrapolate_slightly_larger_k_bf16",
            /*spec=*/
            {/*b=*/1, /*m=*/1024, /*n=*/4096, /*k=*/513, /*clock_cycles=*/0},
            // Expected duration based on (1,1024,4096,512) flops/sec and
            // scaling k.
            /*expected_duration=*/absl::Microseconds(12),
        },
        {
            /*test_name=*/"interpolate_mid_n_bf16",
            /*spec=*/
            {/*b=*/1, /*m=*/1024, /*n=*/2048, /*k=*/512, /*clock_cycles=*/0},
            // Expected duration based on linear interpolation of flops/sec
            // between (1,1024,1024,512) and (1,1024,4096,512).
            /*expected_duration=*/absl::Microseconds(9),
        },
    }),
    [](const TestParamInfo<MatmulInterpolatorDefaultTableTest::ParamType>&
           info) { return info.param.test_name; });

using F8Test = MatmulInterpolatorDefaultTableTest;

TEST_P(F8Test, EstimatesRuntimeForFP8) {
  const auto& [_, spec, expected_duration] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(DotContext context,
                          DotFP8(spec.b, spec.m, spec.n, spec.k));
  // Compare with nanosecond precision.
  EXPECT_EQ(absl::Trunc(*interpolator().EstimatedRuntime(*context.dot),
                        absl::Microseconds(1)),
            expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterpolatorDefaultTableTestInstantiationFP8, F8Test,
    ValuesIn<ParametrizedTestCase>({
        {
            /*test_name=*/"extrapolate_small_fp8",
            /*spec=*/
            {/*b=*/1, /*m=*/64, /*n=*/64, /*k=*/64, /*clock_cycles=*/0},
            // Expected duration based on nearest point (1,512,512,512)
            // flops/sec and scaling by new dimensions.
            /*expected_duration=*/absl::Microseconds(0),
        },
        {
            /*test_name=*/"interpolate_larger_fp8",
            /*spec=*/
            {/*b=*/1, /*m=*/2048, /*n=*/2048, /*k=*/2048, /*clock_cycles=*/0},
            // Expected duration based on nearest point (1,2048,2048,2048)
            // flops/sec and scaling by new dimensions.
            /*expected_duration=*/absl::Microseconds(69),
        },
    }),
    [](const TestParamInfo<MatmulInterpolatorDefaultTableTest::ParamType>&
           info) { return info.param.test_name; });

class MatmulInterpolatorTest : public Test {
 public:
  void SetUp() override {
    constexpr char perf_table[] = R"pb(
      entries {
        instruction {
          opcode: "dot"
          shape {
            element_type: BF16
            dimensions: 1
            dimensions: 1024
            dimensions: 1024
          }
          dot_dimension_numbers {
            lhs_contracting_dimensions: 2
            rhs_contracting_dimensions: 1
            lhs_batch_dimensions: 0
            rhs_batch_dimensions: 0
          }
          id: 2
          operand_ids: 0
          operand_ids: 1
        }
        operands {
          name: "lhs"
          opcode: "parameter"
          shape {
            element_type: BF16
            dimensions: 1
            dimensions: 1024
            dimensions: 1024
          }
        }
        operands {
          name: "rhs"
          opcode: "parameter"
          shape {
            element_type: BF16
            dimensions: 1
            dimensions: 1024
            dimensions: 1024
          }
          parameter_number: 1
          id: 1
        }
        clock_cycles: 1410000000
      }
    )pb";
    HloInstructionProfileList profiles;
    CHECK(tsl::protobuf::TextFormat::ParseFromString(perf_table, &profiles));
    interpolator_ = *MatmulInterpolator::Create(
        profiles, TestGpuDeviceInfo::RTXA6000DeviceInfo());
  }

 protected:
  MatmulInterpolator& interpolator() { return *interpolator_; }

 private:
  std::unique_ptr<MatmulInterpolator> interpolator_;
};

TEST_F(MatmulInterpolatorTest, SupportsCublasCustomCalls) {
  absl::string_view hlo = R"(
    HloModule m

    ENTRY e {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT _ =  (bf16[1024,1024], s8[2097152]{0}) custom-call(p0,p1),
        custom_call_target="__cublas$gemm",
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "gemm_backend_config":{
            "alpha_real":1,
            "beta":1,
            "dot_dimension_numbers": {
              "lhs_contracting_dimensions":["1"],
              "rhs_contracting_dimensions":["1"],
              "lhs_batch_dimensions":[],
              "rhs_batch_dimensions":[]
            }
          }
        }
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  const HloInstruction& custom_call =
      *module->entry_computation()->root_instruction();
  EXPECT_EQ(*interpolator().EstimatedRuntime(custom_call), absl::Seconds(1));
}

TEST_F(MatmulInterpolatorTest, SupportsDotTritonFusion) {
  absl::string_view hlo = R"(
    HloModule m

    comp {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT dot = bf16[1024,1024] dot(p0,p1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
    }

    ENTRY e {
      p0 = bf16[1024,1024] parameter(0)
      p1 = bf16[1024,1024] parameter(1)
      ROOT _ =  bf16[1024,1024] fusion(p0,p1),
        kind=kCustom,
        calls=comp,
        backend_config={
          "operation_queue_id":"0",
          "wait_on_operation_queues":[],
          "fusion_backend_config": {
            "kind":"__triton_gemm",
            "triton_gemm_config":{
              "block_m":"128",
              "block_n":"128",
              "block_k":"64",
              "split_k":"1",
              "num_stages":"1",
              "num_warps":"8",
              "num_ctas":"1"
            }
          },
        }
    }
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnUnverifiedModule(hlo));
  const HloInstruction& custom_call =
      *module->entry_computation()->root_instruction();
  EXPECT_EQ(*interpolator().EstimatedRuntime(custom_call), absl::Seconds(1));
}

}  // namespace
}  // namespace xla::gpu
