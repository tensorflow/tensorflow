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

#include "xla/backends/cpu/runtime/xnnpack/xnn_convolution_thunk.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/error_spec.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

class XnnConvolutionThunkTest
    : public ::testing::TestWithParam<std::tuple<bool, std::vector<int32_t>>> {
 protected:
  bool use_threadpool() const { return std::get<0>(GetParam()); }

  int32_t dimension(int32_t index) const {
    return std::get<1>(GetParam())[index];
  }

  bool IsOdd(int n) { return n % 2 == 1; }
};

TEST_P(XnnConvolutionThunkTest, SimpleConvolution) {
  int32_t batch = dimension(0);
  int32_t height = dimension(1);
  int32_t width = dimension(2);
  int32_t input_channels = dimension(3);
  int32_t kernel_h = dimension(4);
  int32_t kernel_w = dimension(5);
  int32_t output_channels = dimension(6);

  // Padding values for 'SAME' padding. Only odd kernel sizes are supported.
  CHECK(IsOdd(kernel_h) && IsOdd(kernel_w));
  int padding_h = (kernel_h - 1) / 2;
  int padding_w = (kernel_w - 1) / 2;

  std::minstd_rand0 engine;

  // Input format is NHWC.
  auto input_shape =
      ShapeUtil::MakeShape(F32, {batch, height, width, input_channels});

  // Kernel format is HWIO.
  auto kernel_shape = ShapeUtil::MakeShape(
      F32, {kernel_h, kernel_w, input_channels, output_channels});

  auto input =
      *LiteralUtil::CreateRandomLiteral<F32>(input_shape, &engine, 1.0f, 0.1f);
  auto kernel =
      *LiteralUtil::CreateRandomLiteral<F32>(kernel_shape, &engine, 1.0f, 0.1f);

  // Create a reference HLO module that we can use to compare the results.
  std::string hlo_module_template = R"(
    HloModule convolution

    ENTRY TestComputation {
      %p0 = $0 parameter(0)
      %p1 = $1 parameter(1)
      ROOT conv = convolution(p0, p1), window={size=$2 pad=$3},
        dim_labels=b01f_01io->b01f
    }
  )";

  std::string hlo_module = absl::Substitute(
      hlo_module_template, input_shape.ToString(), kernel_shape.ToString(),
      absl::StrCat(kernel_h, "x", kernel_w),
      absl::StrCat(padding_h, "_", padding_h, "x", padding_w, "_", padding_w));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HloModule> module,
      ParseAndReturnUnverifiedModule(hlo_module, HloModuleConfig()));

  HloEvaluator evaluator;
  TF_ASSERT_OK_AND_ASSIGN(Literal expected_result,
                          evaluator.Evaluate(*module, {&input, &kernel}));

  HloInstruction* conv =
      hlo_query::FindInstruction(module->entry_computation(), "conv");
  ASSERT_NE(conv, nullptr);

  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // XNNPACK expects OHWI format for the kernel.
  Literal kernel_transposed =
      kernel.Transpose({3, 0, 1, 2})
          .Relayout(LayoutUtil::MakeLayout({3, 2, 1, 0}));

  // Create a Literal with the expected shape.
  const Shape& out_shape = expected_result.shape();
  auto out = LiteralUtil::CreateFull(out_shape.dimensions(), 0.f);

  BufferAllocations allocations =
      CreateBufferAllocations(input, kernel_transposed, out);

  auto [input_alloc, kernel_transposed_alloc, out_alloc] =
      CreateBufferAllocation(input, kernel_transposed, out);
  auto [input_slice, kernel_transposed_slice, out_slice] =
      CreateBufferAllocationSlice(input_alloc, kernel_transposed_alloc,
                                  out_alloc);

  // Adjust kernel dimensions for XNNPACK.
  ConvolutionDimensionNumbers dnums = conv->convolution_dimension_numbers();
  dnums.set_kernel_input_feature_dimension(3);
  dnums.set_kernel_output_feature_dimension(0);
  dnums.set_kernel_spatial_dimensions(0, 1);
  dnums.set_output_spatial_dimensions(1, 2);

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      XnnConvolutionThunk::Create(
          XnnConvolutionThunk::Options{use_threadpool()}, {"convolution"},
          input_slice, input_shape, kernel_transposed_slice,
          kernel_transposed.shape(), out_slice, out_shape, dnums,
          conv->window(), conv->feature_group_count()));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = use_threadpool() ? &device : nullptr;

  auto execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  ErrorSpec error_spec{1e-5};
  EXPECT_TRUE(LiteralTestUtil::Near(expected_result, out, error_spec));

  // Execute thunk one more time to test that we reuse XNN runtime.
  execute_event = thunk->Execute(params);
  tsl::BlockUntilReady(execute_event);
  ASSERT_FALSE(execute_event.IsError()) << execute_event.GetError();

  EXPECT_TRUE(LiteralTestUtil::Near(expected_result, out, error_spec));
}

INSTANTIATE_TEST_SUITE_P(
    XnnConvolution, XnnConvolutionThunkTest,
    ::testing::Combine(::testing::Values(true, false),
                       ::testing::Values(std::vector<int32_t>{1, 8, 8, 16, 1, 1,
                                                              32})));

}  // namespace
}  // namespace xla::cpu
