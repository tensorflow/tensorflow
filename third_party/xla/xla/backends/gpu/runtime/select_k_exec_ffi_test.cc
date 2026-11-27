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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_registry.h"
#include "xla/ffi/invoke.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_address.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

namespace se = ::stream_executor;

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::TestParamInfo;
using ::testing::TestWithParam;
using ::testing::Values;

constexpr int64_t kMaxInt32 = std::numeric_limits<int32_t>::max();
constexpr int64_t kOverInt32Max = kMaxInt32 + 1;
constexpr int64_t kUint32Wrap = int64_t{1} << 32;

absl::Status InvokeSelectKFfi(PrimitiveType in_dtype,
                              absl::Span<const int64_t> in_dims,
                              PrimitiveType out_dtype,
                              absl::Span<const int64_t> out_dims,
                              absl::Span<const int64_t> indices_dims,
                              int64_t scratch_size = 1024,
                              absl::string_view platform = "CUDA") {
  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(kTopKCustomCallTarget, platform);
  if (!registration.ok()) {
    return registration.status();
  }

  ffi::CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/3);
  builder.AddBufferArg(se::DeviceAddressBase(), in_dtype, in_dims);
  builder.AddBufferRet(se::DeviceAddressBase(), out_dtype, out_dims);
  builder.AddBufferRet(se::DeviceAddressBase(), PrimitiveType::S32,
                       indices_dims);
  builder.AddBufferRet(se::DeviceAddressBase(), PrimitiveType::U8,
                       {scratch_size});
  ffi::CallFrame call_frame = builder.Build();

  ffi::InvokeContext context;
  context.device_ordinal = 0;
  context.backend_context = ffi::InvokeContext::GpuContext{};
  return ffi::Invoke(ffi::GetXlaFfiApi(), registration->bundle.execute,
                     call_frame, context);
}

absl::Status InvokeSelectKFfi(PrimitiveType dtype,
                              absl::Span<const int64_t> in_dims,
                              absl::Span<const int64_t> out_dims,
                              int64_t scratch_size = 1024) {
  return InvokeSelectKFfi(dtype, in_dims, dtype, out_dims, out_dims,
                          scratch_size);
}

TEST(SelectKExecFfiTest, HandlerIsRegisteredForCudaAndRocm) {
  EXPECT_THAT(ffi::FindHandler(kTopKCustomCallTarget, "CUDA"), IsOk());
  EXPECT_THAT(ffi::FindHandler(kTopKCustomCallTarget, "ROCM"), IsOk());
}

TEST(SelectKExecFfiTest, RejectsInvalidRanksAndMismatchedMetadata) {
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {}, {}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("1D or 2D input buffer")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {2, 4, 8}, {2, 4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("1D or 2D input buffer")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {4, 16}, {4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output rank to match input rank")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {4, 16}, PrimitiveType::BF16,
                               {4, 4}, {4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output element type to match")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {4, 16}, {8, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("output batch size to match")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::F32, {4, 16}, PrimitiveType::F32,
                               {4, 4}, {4, 8}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("indices output dimensions to match")));
}

TEST(SelectKExecFfiTest, RejectsNonPositiveScratchBufferSize) {
  EXPECT_THAT(
      InvokeSelectKFfi(PrimitiveType::F32, {4, 16}, {4, 4}, /*scratch_size=*/0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("positive scratch_buffer size")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::U32, {4, 16}, {4, 4},
                               /*scratch_size=*/-1),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("positive scratch_buffer size")));
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::U64, {4, 16}, {4, 4},
                               /*scratch_size=*/
                               std::numeric_limits<int64_t>::min()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("positive scratch_buffer size")));
  // Positive scratch size passes validation and reaches dtype dispatch.
  EXPECT_THAT(
      InvokeSelectKFfi(PrimitiveType::S32, {4, 16}, {4, 4}, /*scratch_size=*/1),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("Unsupported dtype")));
}

struct InvalidDimsCase {
  std::string test_name;
  PrimitiveType dtype;
  std::vector<int64_t> in_dims;
  std::vector<int64_t> out_dims;
  std::string expected_error_substr;
};

class SelectKExecFfiInvalidDimsTest : public TestWithParam<InvalidDimsCase> {};

TEST_P(SelectKExecFfiInvalidDimsTest, RejectsInvalidDimensions) {
  const InvalidDimsCase& test_case = GetParam();
  EXPECT_THAT(
      InvokeSelectKFfi(test_case.dtype, test_case.in_dims, test_case.out_dims),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(test_case.expected_error_substr)));
}

INSTANTIATE_TEST_SUITE_P(
    InvalidDims, SelectKExecFfiInvalidDimsTest,
    Values(
        InvalidDimsCase{"ZeroBatch2D",
                        PrimitiveType::F32,
                        {0, 16},
                        {0, 4},
                        "positive dimensions"},
        InvalidDimsCase{"NegativeBatch2D",
                        PrimitiveType::BF16,
                        {-1, 16},
                        {-1, 4},
                        "positive dimensions"},
        InvalidDimsCase{"ZeroNumElements1D",
                        PrimitiveType::F32,
                        {0},
                        {1},
                        "positive dimensions"},
        InvalidDimsCase{"ZeroNumElements2D",
                        PrimitiveType::U64,
                        {4, 0},
                        {4, 1},
                        "positive dimensions"},
        InvalidDimsCase{"NegativeNumElements1D",
                        PrimitiveType::U32,
                        {-8},
                        {4},
                        "positive dimensions"},
        InvalidDimsCase{
            "ZeroK1D", PrimitiveType::F32, {16}, {0}, "positive dimensions"},
        InvalidDimsCase{"ZeroK2D",
                        PrimitiveType::U32,
                        {4, 16},
                        {4, 0},
                        "positive dimensions"},
        InvalidDimsCase{"NegativeK2D",
                        PrimitiveType::F32,
                        {4, 16},
                        {4, -1},
                        "positive dimensions"},
        InvalidDimsCase{
            "KGreaterThanN1D", PrimitiveType::F32, {16}, {17}, "k <= n"},
        InvalidDimsCase{
            "KGreaterThanN2D", PrimitiveType::U32, {4, 16}, {4, 32}, "k <= n"},
        InvalidDimsCase{"BatchExceedsInt32Max",
                        PrimitiveType::F32,
                        {kOverInt32Max, 16},
                        {kOverInt32Max, 1},
                        "int32_t max"},
        InvalidDimsCase{"BatchWraparoundUint32",
                        PrimitiveType::U64,
                        {kUint32Wrap, 16},
                        {kUint32Wrap, 1},
                        "int32_t max"},
        InvalidDimsCase{"NumElementsExceedsInt32Max1D",
                        PrimitiveType::F32,
                        {kOverInt32Max},
                        {1},
                        "int32_t max"},
        InvalidDimsCase{"NumElementsExceedsInt32Max2D",
                        PrimitiveType::BF16,
                        {1, kOverInt32Max},
                        {1, 1},
                        "int32_t max"},
        InvalidDimsCase{"NumElementsWraparoundUint32",
                        PrimitiveType::U32,
                        {1, kUint32Wrap},
                        {1, 1},
                        "int32_t max"},
        InvalidDimsCase{"BatchTimesKExceedsUint32Max",
                        PrimitiveType::F32,
                        {int64_t{1} << 22, 2048},
                        {int64_t{1} << 22, 1024},
                        "uint32_t max"},
        InvalidDimsCase{"BatchTimesKJustOverUint32Max",
                        PrimitiveType::U32,
                        {65536, 65536},
                        {65536, 65536},
                        "uint32_t max"}),
    [](const TestParamInfo<InvalidDimsCase>& info) {
      return info.param.test_name;
    });

struct ValidBoundaryDimsCase {
  std::string test_name;
  std::vector<int64_t> in_dims;
  std::vector<int64_t> out_dims;
};

class SelectKExecFfiValidBoundaryDimsTest
    : public TestWithParam<ValidBoundaryDimsCase> {};

// Uses an unsupported dtype (S32) so valid boundary dimensions pass all shape
// and dimension checks and reach dtype dispatch without dereferencing a GPU
// stream on either CPU or CUDA builds.
TEST_P(SelectKExecFfiValidBoundaryDimsTest, PassesDimensionValidation) {
  const ValidBoundaryDimsCase& test_case = GetParam();
  EXPECT_THAT(InvokeSelectKFfi(PrimitiveType::S32, test_case.in_dims,
                               test_case.out_dims),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Unsupported dtype")));
}

INSTANTIATE_TEST_SUITE_P(
    ValidBoundaryDims, SelectKExecFfiValidBoundaryDimsTest,
    Values(ValidBoundaryDimsCase{"Valid1D", {16}, {4}},
           ValidBoundaryDimsCase{"Valid2D", {4, 16}, {4, 4}},
           ValidBoundaryDimsCase{"KEqualsN1D", {16}, {16}},
           ValidBoundaryDimsCase{"KEqualsN2D", {4, 16}, {4, 16}},
           ValidBoundaryDimsCase{"SingleElement", {1, 1}, {1, 1}},
           ValidBoundaryDimsCase{
               "BatchAtInt32Max", {kMaxInt32, 4}, {kMaxInt32, 1}},
           ValidBoundaryDimsCase{
               "NumElementsAtInt32Max1D", {kMaxInt32}, {kMaxInt32}},
           ValidBoundaryDimsCase{
               "NumElementsAtInt32Max2D", {1, kMaxInt32}, {1, kMaxInt32}},
           ValidBoundaryDimsCase{
               "BatchTimesKAtUint32Max", {65535, 65537}, {65535, 65537}}),
    [](const TestParamInfo<ValidBoundaryDimsCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::gpu
