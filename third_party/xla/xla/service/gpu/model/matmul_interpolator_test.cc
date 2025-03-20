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
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

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

class MatmulInterpolatorTest : public TestWithParam<ParametrizedTestCase> {
 public:
  MatmulInterpolatorTest()
      : device_info_(TestGpuDeviceInfo::RTXA6000DeviceInfo()) {}

  void SetUp() override {
    absl::StatusOr<HloInstructionProfileList> profiles =
        DotInterpolationSpace(interpolation_space_);
    CHECK_OK(profiles.status()) << "Cannot generate interpolation space.";
    absl::StatusOr<std::unique_ptr<MatmulInterpolator>> interpolator =
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

  MatmulInterpolator& interpolator() { return *interpolator_; }

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
  std::unique_ptr<MatmulInterpolator> interpolator_;
};

TEST_P(MatmulInterpolatorTest, MatmulInteprolatorNextNeighbourInterpolation) {
  const auto& [_, spec, expected_duration] = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(DotContext context,
                          Dot(spec.b, spec.m, spec.n, spec.k));
  EXPECT_EQ(absl::Trunc(*interpolator().EstimatedRuntime(*context.dot),
                        absl::Milliseconds(1)),
            expected_duration);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulInterpolatorTestInstantiation, MatmulInterpolatorTest,
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
    [](const TestParamInfo<MatmulInterpolatorTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::gpu
