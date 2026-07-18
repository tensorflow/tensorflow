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

#include "xla/backends/cpu/runtime/onednn/onednn_op_thunk.h"

#include <vector>

// #include "gtest/gtest.h"
#include "xla/array2d.h"
#include "xla/backends/cpu/runtime/buffer_allocations.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk_testlib.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {
namespace {

TEST(OneDnnOpThunkTest, SimpleOneDnnMatMulThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Define shapes for lhs (2x3), rhs (3x2), and output (2x2)
  Shape lhs_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape rhs_shape = ShapeUtil::MakeShape(F32, {3, 2});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 2});

  // Prepare dummy data (row-major):
  // A = [ [1, 2, 3],
  //       [4, 5, 6] ]
  // B = [ [7,  8],
  //       [9, 10],
  //       [11,12] ]
  // C = A * B =
  //     [ [58, 64],
  //       [139,154] ]
  std::vector<float> input_a = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> input_b = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
  std::vector<float> output(4, 0.f);

  // Create Literals from data
  Literal lhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal rhs_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{7.f, 8.f}, {9.f, 10.f}, {11.f, 12.f}}));
  Literal out_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f}, {0.f, 0.f}}));

  // Create buffer allocations and slices
  auto [lhs_alloc, rhs_alloc, out_alloc] =
      CreateBufferAllocation(lhs_literal, rhs_literal, out_literal);

  auto [lhs_slice, rhs_slice, out_slice] =
      CreateBufferAllocationSlice(lhs_alloc, rhs_alloc, out_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(lhs_literal, rhs_literal, out_literal);

  // Set up op_buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {lhs_slice, rhs_slice};
  op_buffers.arguments_shapes = {lhs_shape, rhs_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  // Create thunk (matmul)
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk,
      OneDnnOpThunk::Create("__onednn$matmul", Thunk::Info(), op_buffers, {}));

  // Set up execute params
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  // Execute the thunk
  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "OneDnnOpThunk execution failed";

  // Expected output literal
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{58.f, 64.f}, {139.f, 154.f}}));

  // Load output and verify
  EXPECT_EQ(out_literal, expected);
}

// Small 2D NHWC convolution test for OneDnnOpThunk.
// Input: 1x3x3x1, Kernel: 2x2x1x1 with values [[1,0],[0,1]] (HWIO).
// Output (valid conv): each element = top-left + bottom-right of 2x2 patch:
// [[1+5, 2+6], [4+8, 5+9]] = [[6, 8], [12, 14]].
TEST(OneDnnOpThunkTest, SimpleOneDnnConvolutionThunk) {
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 4);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Input: N=1,H=3,W=3,C=1 (NHWC)
  // Weights: KH=2, KW=2, IC=1, OC=1 (HWIO)
  // Output: N=1,H=2,W=2,C=1 (stride=1, no pad)
  Shape input_shape = ShapeUtil::MakeShape(F32, {1, 3, 3, 1});
  Shape weight_shape = ShapeUtil::MakeShape(F32, {2, 2, 1, 1});
  Shape output_shape = ShapeUtil::MakeShape(F32, {1, 2, 2, 1});

  // Input data (H x W):
  // 1 2 3
  // 4 5 6
  // 7 8 9
  Literal input_literal =
      LiteralUtil::CreateR4FromArray4D<float>(Array4D<float>(
          1, 3, 3, 1, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}));

  // 2x2 kernel:
  // 1 0
  // 0 1
  Literal weight_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(2, 2, 1, 1, {1.f, 0.f, 0.f, 1.f}));

  // Output buffer init zeros
  Literal output_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 2, 2, 1, {0.f, 0.f, 0.f, 0.f}));

  // Expected convolution (valid):
  // [[ (1*1 +2*0 +4*0 +5*1)=6,  (2*1 +3*0 +5*0 +6*1)=8 ],
  //  [ (4*1 +5*0 +7*0 +8*1)=12, (5*1 +6*0 +8*0 +9*1)=14 ]]
  Literal expected_literal = LiteralUtil::CreateR4FromArray4D<float>(
      Array4D<float>(1, 2, 2, 1, {6.f, 8.f, 12.f, 14.f}));

  // Buffer allocations
  auto [input_alloc, weight_alloc, output_alloc] =
      CreateBufferAllocation(input_literal, weight_literal, output_literal);

  auto [input_slice, weight_slice, output_slice] =
      CreateBufferAllocationSlice(input_alloc, weight_alloc, output_alloc);

  BufferAllocations allocations =
      CreateBufferAllocations(input_literal, weight_literal, output_literal);

  // Build a minimal OneDnnConvolutionConfig proto.
  OneDnnConvolutionConfig conv_config;
  conv_config.set_dims(4);
  conv_config.set_feature_groups(1);

  OneDnnTensorLayoutProto* inp = conv_config.mutable_input();
  inp->set_dims(4);
  OneDnnDataLayoutProto* inp_data = inp->mutable_data();
  inp_data->set_batch_dim(0);
  inp_data->set_feature_dim(3);
  inp_data->add_spatial_dims(1);
  inp_data->add_spatial_dims(2);

  // Kernel layout assumed HWIO (H,W,In,Out):
  OneDnnTensorLayoutProto* ker = conv_config.mutable_kernel();
  ker->set_dims(4);
  OneDnnFilterLayoutProto* filter = ker->mutable_filter();
  filter->set_input_feature_dim(2);
  filter->set_output_feature_dim(3);
  filter->add_spatial_dims(0);
  filter->add_spatial_dims(1);

  // Output layout NHWC
  OneDnnTensorLayoutProto* out = conv_config.mutable_output();
  out->set_dims(4);
  OneDnnDataLayoutProto* out_data = out->mutable_data();
  out_data->set_batch_dim(0);
  out_data->set_feature_dim(3);
  out_data->add_spatial_dims(1);
  out_data->add_spatial_dims(2);

  conv_config.set_feature_groups(1);

  // Window parameters: stride=1, pad=0, dilation=1.
  OneDnnWindowProto* win = conv_config.mutable_window();
  win->add_strides(1);
  win->add_strides(1);
  win->add_pad_left(0);
  win->add_pad_left(0);
  win->add_pad_right(0);
  win->add_pad_right(0);
  win->add_window_dilations(1);
  win->add_window_dilations(1);

  // Set up op buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {input_slice, weight_slice};
  op_buffers.arguments_shapes = {input_shape, weight_shape};
  op_buffers.results_buffers = {output_slice};
  op_buffers.results_shapes = {output_shape};

  // Wrap config in variant
  OneDnnOpThunk::OneDnnOpConfig config_variant = conv_config;

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$convolution", Thunk::Info(),
                                        op_buffers, config_variant));

  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError())
      << "OneDnnOpThunk convolution execution failed";

  EXPECT_EQ(output_literal, expected_literal);
}

TEST(OneDnnOpThunkTest, SimpleOneDnnLayerNormThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Shapes: input [2,3], gamma [3], beta [3], output [2,3]
  Shape input_shape = ShapeUtil::MakeShape(F32, {2, 3});
  Shape gamma_shape = ShapeUtil::MakeShape(F32, {3});
  Shape beta_shape = ShapeUtil::MakeShape(F32, {3});
  Shape out_shape = ShapeUtil::MakeShape(F32, {2, 3});

  // Input:
  // [[1,2,3],
  //  [4,5,6]]
  // gamma = [1,1,1]
  // beta  = [0,0,0]
  // Expected (per-row LN): [[-1.2247449, 0, 1.2247449],
  //                         [-1.2247449, 0, 1.2247449]]
  Literal input_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));
  Literal gamma_literal = LiteralUtil::CreateR1<float>({1.f, 1.f, 1.f});
  Literal beta_literal = LiteralUtil::CreateR1<float>({0.f, 0.f, 0.f});
  Literal out_literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}}));

  // Buffer allocations
  auto [input_alloc, gamma_alloc, beta_alloc, out_alloc] =
      CreateBufferAllocation(input_literal, gamma_literal, beta_literal,
                             out_literal);

  auto [input_slice, gamma_slice, beta_slice, out_slice] =
      CreateBufferAllocationSlice(input_alloc, gamma_alloc, beta_alloc,
                                  out_alloc);

  BufferAllocations allocations = CreateBufferAllocations(
      input_literal, gamma_literal, beta_literal, out_literal);

  // Op buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {input_slice, gamma_slice, beta_slice};
  op_buffers.arguments_shapes = {input_shape, gamma_shape, beta_shape};
  op_buffers.results_buffers = {out_slice};
  op_buffers.results_shapes = {out_shape};

  // oneDNN LayerNorm config
  OneDnnNormConfig ln_cfg;
  ln_cfg.set_rescale(OneDnnNormConfig::SCALE_AND_SHIFT);
  float epsilon = 1e-5f;
  int32_t epsilon_bits = *reinterpret_cast<int32_t*>(&epsilon);
  ln_cfg.set_epsilon_typecast(epsilon_bits);

  OneDnnOpThunk::OneDnnOpConfig config = ln_cfg;

  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$layernorm", Thunk::Info(),
                                        op_buffers, config));

  // Execute
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocations;
  params.intra_op_threadpool = &device;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError()) << "oneDNN LayerNorm thunk failed";

  // Expected output
  const float ref = 1.2247449f;  // sqrt(3/2)
  Literal expected = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{-ref, 0.f, ref}, {-ref, 0.f, ref}}));

  // Compare with tolerance
  const float* exp_view = expected.data<float>().data();
  float* out_view = out_literal.data<float>().data();
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(out_view[i], exp_view[i], 1e-4)
        << "Mismatch at index " << i << ": got " << out_view[i] << " expected "
        << exp_view[i];
  }
}

TEST(OneDnnOpThunkTest, SimpleOneDnnSoftmaxThunk) {
  // Set up a thread pool for parallel execution
  tsl::thread::ThreadPool threads(tsl::Env::Default(), "test", 8);
  Eigen::ThreadPoolDevice device(threads.AsEigenThreadPool(),
                                 threads.NumThreads());

  // Input shape (2x3), softmax over axis=1 (last dim)
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});

  // Input:
  // [[1,2,3],
  //  [4,5,6]]
  Literal literal = LiteralUtil::CreateR2FromArray2D<float>(
      Array2D<float>({{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}}));

  // Buffer allocation
  auto alloc = CreateBufferAllocation(0, literal);
  auto slice = CreateBufferAllocationSlice(alloc);

  BufferAllocations allocation = CreateBufferAllocations(literal);

  // Set up op_buffers
  OneDnnOpThunk::OpBuffers op_buffers;
  op_buffers.arguments_buffers = {slice};
  op_buffers.arguments_shapes = {shape};
  op_buffers.results_buffers = {slice};
  op_buffers.results_shapes = {shape};

  // Softmax config (axis = 1)
  OneDnnSoftmaxConfig softmax_cfg;
  softmax_cfg.set_softmax_axis(1);
  OneDnnOpThunk::OneDnnOpConfig variant_cfg = softmax_cfg;

  // Create thunk for Softmax
  TF_ASSERT_OK_AND_ASSIGN(
      auto thunk, OneDnnOpThunk::Create("__onednn$softmax", Thunk::Info(),
                                        op_buffers, variant_cfg));

  // Execute params
  Thunk::ExecuteParams params;
  params.buffer_allocations = &allocation;
  params.intra_op_threadpool = &device;

  tsl::AsyncValueRef<Thunk::ExecuteEvent> exec_event = thunk->Execute(params);
  tsl::BlockUntilReady(exec_event);
  ASSERT_FALSE(exec_event.IsError())
      << "OneDnnOpThunk softmax execution failed";

  // Compute expected softmax row-wise
  auto softmax_row = [](float a, float b, float c) {
    float ea = std::exp(a), eb = std::exp(b), ec = std::exp(c);
    float s = ea + eb + ec;
    return std::array<float, 3>{ea / s, eb / s, ec / s};
  };
  std::array<float, 3> r0 = softmax_row(1.f, 2.f, 3.f);
  std::array<float, 3> r1 = softmax_row(4.f, 5.f, 6.f);

  const float kTol = 1e-5f;
  // Validate results
  for (int i = 0; i < 3; ++i) {
    float got0 = literal.Get<float>({0, i});
    float got1 = literal.Get<float>({1, i});
    EXPECT_NEAR(got0, r0[i], kTol);
    EXPECT_NEAR(got1, r1[i], kTol);
  }
}

}  // namespace
}  // namespace xla::cpu
