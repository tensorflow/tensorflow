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

// Verifies that oneDNN's ONEDNN_VERBOSE environment variable is functional.

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

using ::testing::HasSubstr;

// Initialized at namespace scope, i.e. before main(), because oneDNN caches the
// parsed value on the first call that consults it.
const bool kVerboseSet =
    tsl::setenv("ONEDNN_VERBOSE", "1", /*overwrite=*/1) == 0;

// Errors are returned rather than asserted so that stdout capture is restored
// before any failure is reported.
absl::Status ExecuteOneDnnMatMul() {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "onednn-verbose", 2);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 2});

  Literal lhs = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal rhs = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}}));
  Literal out = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f}, {0.f, 0.f}}));

  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs, rhs, out);
  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);
  BufferAllocations allocations = CreateBufferAllocations(lhs, rhs, out);

  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {lhs_slice, rhs_slice};
  op_buffers.arguments_shapes = {lhs_shape, rhs_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<OneDnnOpThunk> thunk,
      OneDnnOpThunk::Create("__onednn$matmul", Thunk::Info(), op_buffers, {}));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  if (exec_event.IsError()) return exec_event.GetError();
  return absl::OkStatus();
}

TEST(OneDnnVerboseTest, VerboseEnvVarProducesLogs) {
  ASSERT_TRUE(kVerboseSet) << "failed to set ONEDNN_VERBOSE";

  // oneDNN logs with printf, so capture stdout at the file descriptor level.
  testing::internal::CaptureStdout();
  absl::Status status = ExecuteOneDnnMatMul();
  std::string logs = testing::internal::GetCapturedStdout();

  ASSERT_TRUE(status.ok()) << status;

  // Prefix shared by every verbose line.
  EXPECT_THAT(logs, HasSubstr("onednn_verbose"));

  // ONEDNN_VERBOSE=1 is verbose_t::level1, which includes exec_profile.
  EXPECT_THAT(logs, HasSubstr("primitive,exec"));
}

}  // namespace
}  // namespace xla::cpu
