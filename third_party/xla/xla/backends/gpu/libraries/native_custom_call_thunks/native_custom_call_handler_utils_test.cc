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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/libraries/native_custom_call_thunks/native_custom_call_handler_testlib.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/kernel_args_packing_spec.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::SizeIs;

namespace se = ::stream_executor;

constexpr absl::string_view kArrayResultHlo = R"(
  ENTRY e {
    p0 = f32[4] parameter(0)
    ROOT c = f32[4] custom-call(p0), custom_call_target="test.target"
  }
)";

constexpr absl::string_view kTupleResultHlo = R"(
  ENTRY e {
    p0 = f32[4] parameter(0)
    ROOT c = (f32[4]) custom-call(p0), custom_call_target="test.target"
  }
)";

// A kernel spec that can be built without a GPU: the PTX is never loaded
// because the tests only inspect the emitted thunk.
se::KernelLoaderSpec FakeKernelSpec(int arity) {
  return se::KernelLoaderSpec::CreateOwningCudaPtxInMemorySpec(
      /*ptx=*/"", /*kernel_name=*/"fake_kernel", arity);
}

TEST(SingleResultShapeIndexTest, ArrayResult) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kArrayResultHlo));
  EXPECT_THAT(SingleResultShapeIndex(tester->instruction()),
              IsOkAndHolds(ShapeIndex{}));
}

TEST(SingleResultShapeIndexTest, ResultWrappedInAOneElementTuple) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kTupleResultHlo));
  EXPECT_THAT(SingleResultShapeIndex(tester->instruction()),
              IsOkAndHolds(ShapeIndex{0}));
}

TEST(SingleResultShapeIndexTest, RejectsMultipleResults) {
  ASSERT_OK_AND_ASSIGN(auto tester, NativeCustomCallHandlerTester::Create(R"(
    ENTRY e {
      p0 = f32[4] parameter(0)
      ROOT c = (f32[4], f32[4]) custom-call(p0), custom_call_target="t"
    }
  )"));
  EXPECT_THAT(SingleResultShapeIndex(tester->instruction()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("exactly one array result")));
}

TEST(GetSingleResultShapedSliceTest, ReturnsShapeAndSliceOfTupleElement) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(kTupleResultHlo));
  ASSERT_OK_AND_ASSIGN(
      ShapedSlice result,
      GetSingleResultShapedSlice(tester->instruction(), tester->context()));

  EXPECT_EQ(result.shape, ShapeUtil::MakeShape(F32, {4}));
  EXPECT_EQ(result.slice.size(), 4 * sizeof(float));
}

TEST(GetCudaComputeCapabilityTest, ReportsTheTargetCapability) {
  ASSERT_OK_AND_ASSIGN(
      auto tester, NativeCustomCallHandlerTester::Create(
                       kArrayResultHlo, {/*gpu_model=*/GpuModel::H100_SXM}));
  ASSERT_OK_AND_ASSIGN(se::CudaComputeCapability compute_capability,
                       GetCudaComputeCapability(tester->context()));
  // The H100 target config compiles for sm_90a, i.e. it opts into features
  // that only run on a device of exactly this compute capability.
  EXPECT_EQ(compute_capability, se::CudaComputeCapability::H100Accelerated());
}

TEST(GetCudaComputeCapabilityTest, FailsOnNonCudaTarget) {
  ASSERT_OK_AND_ASSIGN(auto tester,
                       NativeCustomCallHandlerTester::Create(
                           kArrayResultHlo, {/*gpu_model=*/GpuModel::MI200}));
  EXPECT_THAT(GetCudaComputeCapability(tester->context()),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("CUDA device")));
}

class MakeCustomKernelThunkSequenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        tester_, NativeCustomCallHandlerTester::Create(kArrayResultHlo));
    // One operand plus one result.
    ASSERT_OK_AND_ASSIGN(kernel_args_,
                         tester_->context().CreateKernelArguments());
    ASSERT_THAT(kernel_args_->args(), SizeIs(2));
  }

  // The packing spec a correct handler would build for this custom call.
  se::KernelArgsPackingSpec ValidPackingSpec() {
    se::KernelArgsPackingSpec spec;
    spec.AddAddressArgument(0);
    spec.AddAddressArgument(1);
    return spec;
  }

  // The launch spec a correct handler would build for this custom call.
  CustomKernelLaunchSpec ValidLaunchSpec() {
    return CustomKernelLaunchSpec{/*name=*/"fake_kernel",
                                  /*kernel_spec=*/FakeKernelSpec(2),
                                  /*packing_spec=*/ValidPackingSpec(),
                                  /*block_dims=*/se::BlockDim(4)};
  }

  std::unique_ptr<NativeCustomCallHandlerTester> tester_;
  std::optional<emitters::KernelArguments> kernel_args_;
};

TEST_F(MakeCustomKernelThunkSequenceTest, BuildsASingleThunk) {
  ASSERT_OK_AND_ASSIGN(
      ThunkSequence thunks,
      MakeCustomKernelThunkSequence(tester_->context(), ValidLaunchSpec(),
                                    *kernel_args_));

  ASSERT_THAT(thunks, SizeIs(1));
  EXPECT_EQ(thunks[0]->kind(), Thunk::kCustomKernel);
  // Programmatic Dependent Launch is off unless a handler asks for it.
  EXPECT_FALSE(static_cast<CustomKernelThunk&>(*thunks[0]).use_pdl());
}

TEST_F(MakeCustomKernelThunkSequenceTest, ForwardsUsePdl) {
  CustomKernelLaunchSpec spec = ValidLaunchSpec();
  spec.use_pdl = true;

  ASSERT_OK_AND_ASSIGN(ThunkSequence thunks,
                       MakeCustomKernelThunkSequence(
                           tester_->context(), std::move(spec), *kernel_args_));

  ASSERT_THAT(thunks, SizeIs(1));
  EXPECT_TRUE(static_cast<CustomKernelThunk&>(*thunks[0]).use_pdl());
}

TEST_F(MakeCustomKernelThunkSequenceTest, RejectsOutOfRangeRelocation) {
  se::KernelArgsPackingSpec packing_spec;
  packing_spec.AddAddressArgument(0);
  // There are only two kernel arguments, so index 7 cannot be resolved.
  packing_spec.AddAddressArgument(7);

  EXPECT_THAT(
      MakeCustomKernelThunkSequence(tester_->context(),
                                    {/*name=*/"fake_kernel",
                                     /*kernel_spec=*/FakeKernelSpec(2),
                                     /*packing_spec=*/std::move(packing_spec)},
                                    *kernel_args_),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(MakeCustomKernelThunkSequenceTest, RejectsArityMismatch) {
  EXPECT_THAT(
      MakeCustomKernelThunkSequence(tester_->context(),
                                    {/*name=*/"fake_kernel",
                                     // The kernel takes three arguments but
                                     // the packing spec only produces two.
                                     /*kernel_spec=*/FakeKernelSpec(3),
                                     /*packing_spec=*/ValidPackingSpec()},
                                    *kernel_args_),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("takes 3 arguments")));
}

TEST_F(MakeCustomKernelThunkSequenceTest, RejectsOutOfRangeZeroedBuffer) {
  CustomKernelLaunchSpec spec = ValidLaunchSpec();
  spec.zeroed_output_buffer_indices = {5};

  EXPECT_THAT(MakeCustomKernelThunkSequence(tester_->context(), std::move(spec),
                                            *kernel_args_),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Zeroed output buffer index 5")));
}

}  // namespace
}  // namespace xla::gpu
