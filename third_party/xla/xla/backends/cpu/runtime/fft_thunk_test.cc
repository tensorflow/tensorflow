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

#include "xla/backends/cpu/runtime/fft_thunk.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/synchronization/blocking_counter.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(FftThunkTest, ConcurrentExecuteDoesNotDeadlock) {
  constexpr int kNumThreads = 2;
  constexpr int kNumConcurrentCalls = 4;
  constexpr int kIterations = 50;

  tsl::thread::ThreadPool pool(tsl::Env::Default(), "fft_test", kNumThreads);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  // DUCC only parallelizes FFTs when total element count >= 32768.
  Shape shape = ShapeUtil::MakeShape(C64, {128, 256});
  std::vector<int64_t> fft_length = {256};

  struct CallState {
    Literal input;
    Literal output;
  };

  std::vector<CallState> calls(kNumConcurrentCalls);
  for (int call = 0; call < kNumConcurrentCalls; ++call) {
    calls[call].input = LiteralUtil::CreateFullWithDescendingLayout<complex64>(
        {128, 256}, complex64(1.0f, 0.0f));
    calls[call].output = LiteralUtil::CreateFullWithDescendingLayout<complex64>(
        {128, 256}, complex64(0.0f, 0.0f));
  }

  auto [input_alloc, output_alloc] =
      CreateBufferAllocation(calls[0].input, calls[0].output);
  auto [input_slice, output_slice] =
      CreateBufferAllocationSlice(input_alloc, output_alloc);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FftThunk> thunk,
      FftThunk::Create({"fft"}, /*is_multi_thread_eigen=*/true,
                       /*fft_type=*/FftType::FFT, fft_length, input_slice,
                       shape, output_slice, shape));

  for (int iter = 0; iter < kIterations; ++iter) {
    absl::BlockingCounter done(kNumConcurrentCalls);
    for (int call = 0; call < kNumConcurrentCalls; ++call) {
      pool.Schedule([&device, &thunk, &state = calls[call], &done]() {
        BufferAllocations allocations =
            CreateBufferAllocations(state.input, state.output);
        Thunk::ExecuteParams params;
        params.buffer_allocations = &allocations;
        params.intra_op_threadpool = &device;

        auto execute_event = thunk->Execute(params);
        tsl::BlockUntilReady(execute_event);
        EXPECT_FALSE(execute_event.IsError()) << execute_event.GetError();
        done.DecrementCount();
      });
    }
    done.Wait();
  }
}

}  // namespace
}  // namespace xla::cpu
