/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"

#include <algorithm>
#include <cmath>

#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/gpu/partition_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/kernel.h"
#include "tensorflow/stream_executor/stream_executor_pimpl.h"

namespace xla {
namespace gpu {

static constexpr double kTolerance = 0.1f;

// Comparison kernel code: compare two buffers of fp16/fp32/fp64 of length
// buffer_length where the relative error does not exceed the passed
// rel_error_threshold. Write the number of mismatches into out parameter
// mismatch_count.
//
// NaN's are considered equal, and for half's we clamp all numbers to largest
// and smallest numbers representable to avoid miscomparisons due to overflows.
//
// The PTX below is compiled from the following CUDA code:
//
// #include<cuda_fp16.h>
// extern "C" { // avoid name mangling
// __device__ float canonicalize(float input) {
//   // All fp16 infinities are treated as 65505 or -65505, in order to avoid
//   // differences due to overflows.
//   return isnan(input) ? input : max(-65505.0f, min(input, 65505.0f));
// }
//
// __global__ void __xla_fp16_comparison(__half* buffer_a, __half* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = __half2float(buffer_a[idx]);
//   float elem_b = __half2float(buffer_b[idx]);
//   elem_a = canonicalize(elem_a);
//   elem_b = canonicalize(elem_b);
//   if (isnan(elem_a) && isnan(elem_b)) return;
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp32_comparison(float* buffer_a, float* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   float elem_a = buffer_a[idx];
//   float elem_b = buffer_b[idx];
//   if (isnan(elem_a) && isnan(elem_b)) return;
//   if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
//     return;
//   float rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
//
// __global__ void __xla_fp64_comparison(double* buffer_a, double* buffer_b,
//                                       float rel_error_threshold,
//                                       unsigned long long buffer_length,
//                                       int* mismatch_count) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= buffer_length) return;
//   double elem_a = buffer_a[idx];
//   double elem_b = buffer_b[idx];
//   if (isnan(elem_a) && isnan(elem_b)) return;
//   if (isinf(elem_a) && isinf(elem_b) && signbit(elem_a) == signbit(elem_b))
//     return;
//   double rel_error = abs(elem_a - elem_b)
//       / (max(abs(elem_a), abs(elem_b)) + 1);
//   if (rel_error > rel_error_threshold || isnan(rel_error))
//     atomicAdd(mismatch_count, 1);
// }
// } // end extern declaration.
static const char* buffer_compare_ptx = R"(
.version 4.2
.target sm_30
.address_size 64

.visible .entry __xla_fp16_comparison(
  .param .u64 __xla_fp16_comparison_param_0,
  .param .u64 __xla_fp16_comparison_param_1,
  .param .f32 __xla_fp16_comparison_param_2,
  .param .u64 __xla_fp16_comparison_param_3,
  .param .u64 __xla_fp16_comparison_param_4
)
{
  .reg .pred   %p<10>;
  .reg .b16   %rs<3>;
  .reg .f32   %f<20>;
  .reg .b32   %r<6>;
  .reg .b64   %rd<12>;
  ld.param.u64   %rd8, [__xla_fp16_comparison_param_3];
  mov.u32   %r1, %tid.x;
  mov.u32   %r2, %ctaid.x;
  mov.u32   %r3, %ntid.x;
  mad.lo.s32   %r4, %r3, %r2, %r1;
  cvt.s64.s32   %rd4, %r4;
  setp.ge.u64   %p1, %rd4, %rd8;
  @%p1 bra   LBB7_4;
  ld.param.u64   %rd5, [__xla_fp16_comparison_param_0];
  ld.param.u64   %rd7, [__xla_fp16_comparison_param_1];
  cvta.to.global.u64   %rd2, %rd7;
  cvta.to.global.u64   %rd3, %rd5;
  shl.b64   %rd9, %rd4, 1;
  add.s64   %rd10, %rd3, %rd9;
  ld.global.u16   %rs1, [%rd10];
  // begin inline asm
  {  cvt.f32.f16 %f6, %rs1;}

  // end inline asm
  add.s64   %rd11, %rd2, %rd9;
  ld.global.u16   %rs2, [%rd11];
  // begin inline asm
  {  cvt.f32.f16 %f7, %rs2;}

  // end inline asm
  abs.f32   %f8, %f6;
  setp.gtu.f32   %p2, %f8, 0f7F800000;
  min.f32   %f9, %f6, 0f477FE100;
  max.f32   %f10, %f9, 0fC77FE100;
  selp.f32   %f1, %f6, %f10, %p2;
  abs.f32   %f11, %f7;
  setp.gtu.f32   %p3, %f11, 0f7F800000;
  min.f32   %f12, %f7, 0f477FE100;
  max.f32   %f13, %f12, 0fC77FE100;
  selp.f32   %f2, %f7, %f13, %p3;
  abs.f32   %f3, %f1;
  setp.gtu.f32   %p4, %f3, 0f7F800000;
  abs.f32   %f4, %f2;
  setp.gtu.f32   %p5, %f4, 0f7F800000;
  and.pred    %p6, %p4, %p5;
  @%p6 bra   LBB7_4;
  ld.param.f32   %f5, [__xla_fp16_comparison_param_2];
  sub.f32   %f14, %f1, %f2;
  abs.f32   %f15, %f14;
  max.f32   %f16, %f3, %f4;
  add.f32   %f17, %f16, 0f3F800000;
  div.rn.f32   %f18, %f15, %f17;
  setp.leu.f32   %p7, %f18, %f5;
  abs.f32   %f19, %f18;
  setp.le.f32   %p8, %f19, 0f7F800000;
  and.pred    %p9, %p7, %p8;
  @%p9 bra   LBB7_4;
  ld.param.u64   %rd6, [__xla_fp16_comparison_param_4];
  cvta.to.global.u64   %rd1, %rd6;
  atom.global.add.u32   %r5, [%rd1], 1;
LBB7_4:
  ret;

}
  // .globl  __xla_fp32_comparison
.visible .entry __xla_fp32_comparison(
  .param .u64 __xla_fp32_comparison_param_0,
  .param .u64 __xla_fp32_comparison_param_1,
  .param .f32 __xla_fp32_comparison_param_2,
  .param .u64 __xla_fp32_comparison_param_3,
  .param .u64 __xla_fp32_comparison_param_4
)
{
  .reg .pred   %p<12>;
  .reg .f32   %f<12>;
  .reg .b32   %r<9>;
  .reg .b64   %rd<12>;

  ld.param.u64   %rd8, [__xla_fp32_comparison_param_3];
  mov.u32   %r1, %tid.x;
  mov.u32   %r2, %ctaid.x;
  mov.u32   %r3, %ntid.x;
  mad.lo.s32   %r4, %r3, %r2, %r1;
  cvt.s64.s32   %rd4, %r4;
  setp.ge.u64   %p1, %rd4, %rd8;
  @%p1 bra   LBB8_6;
  ld.param.u64   %rd5, [__xla_fp32_comparison_param_0];
  ld.param.u64   %rd7, [__xla_fp32_comparison_param_1];
  cvta.to.global.u64   %rd2, %rd7;
  cvta.to.global.u64   %rd3, %rd5;
  shl.b64   %rd9, %rd4, 2;
  add.s64   %rd10, %rd3, %rd9;
  ld.global.f32   %f1, [%rd10];
  add.s64   %rd11, %rd2, %rd9;
  ld.global.f32   %f2, [%rd11];
  abs.f32   %f3, %f1;
  setp.gtu.f32   %p2, %f3, 0f7F800000;
  abs.f32   %f4, %f2;
  setp.gtu.f32   %p3, %f4, 0f7F800000;
  and.pred    %p4, %p2, %p3;
  @%p4 bra   LBB8_6;
  setp.neu.f32   %p5, %f3, 0f7F800000;
  setp.neu.f32   %p6, %f4, 0f7F800000;
  or.pred    %p7, %p5, %p6;
  @%p7 bra   LBB8_4;
  mov.b32   %r5, %f1;
  mov.b32   %r6, %f2;
  xor.b32    %r7, %r6, %r5;
  setp.gt.s32   %p8, %r7, -1;
  @%p8 bra   LBB8_6;
LBB8_4:
  ld.param.f32   %f5, [__xla_fp32_comparison_param_2];
  sub.f32   %f6, %f1, %f2;
  abs.f32   %f7, %f6;
  max.f32   %f8, %f3, %f4;
  add.f32   %f9, %f8, 0f3F800000;
  div.rn.f32   %f10, %f7, %f9;
  setp.leu.f32   %p9, %f10, %f5;
  abs.f32   %f11, %f10;
  setp.le.f32   %p10, %f11, 0f7F800000;
  and.pred    %p11, %p9, %p10;
  @%p11 bra   LBB8_6;
  ld.param.u64   %rd6, [__xla_fp32_comparison_param_4];
  cvta.to.global.u64   %rd1, %rd6;
  atom.global.add.u32   %r8, [%rd1], 1;
LBB8_6:
  ret;

}
  // .globl  __xla_fp64_comparison
.visible .entry __xla_fp64_comparison(
  .param .u64 __xla_fp64_comparison_param_0,
  .param .u64 __xla_fp64_comparison_param_1,
  .param .f32 __xla_fp64_comparison_param_2,
  .param .u64 __xla_fp64_comparison_param_3,
  .param .u64 __xla_fp64_comparison_param_4
)
{
  .reg .pred   %p<16>;
  .reg .f32   %f<2>;
  .reg .b32   %r<13>;
  .reg .f64   %fd<12>;
  .reg .b64   %rd<12>;

  ld.param.u64   %rd8, [__xla_fp64_comparison_param_3];
  mov.u32   %r2, %tid.x;
  mov.u32   %r3, %ctaid.x;
  mov.u32   %r4, %ntid.x;
  mad.lo.s32   %r5, %r4, %r3, %r2;
  cvt.s64.s32   %rd4, %r5;
  setp.ge.u64   %p1, %rd4, %rd8;
  @%p1 bra   LBB9_6;
  ld.param.u64   %rd5, [__xla_fp64_comparison_param_0];
  ld.param.u64   %rd7, [__xla_fp64_comparison_param_1];
  cvta.to.global.u64   %rd2, %rd7;
  cvta.to.global.u64   %rd3, %rd5;
  shl.b64   %rd9, %rd4, 3;
  add.s64   %rd10, %rd3, %rd9;
  ld.global.f64   %fd1, [%rd10];
  add.s64   %rd11, %rd2, %rd9;
  ld.global.f64   %fd2, [%rd11];
  abs.f64   %fd3, %fd1;
  setp.gtu.f64   %p2, %fd3, 0d7FF0000000000000;
  abs.f64   %fd4, %fd2;
  setp.gtu.f64   %p3, %fd4, 0d7FF0000000000000;
  and.pred    %p4, %p2, %p3;
  @%p4 bra   LBB9_6;
  {
  .reg .b32 %temp; 
  mov.b64   {%r6, %temp}, %fd1;
  }
  {
  .reg .b32 %temp; 
  mov.b64   {%temp, %r1}, %fd1;
  }
  and.b32    %r7, %r1, 2147483647;
  setp.ne.s32   %p5, %r7, 2146435072;
  setp.ne.s32   %p6, %r6, 0;
  or.pred    %p7, %p6, %p5;
  @%p7 bra   LBB9_4;
  {
  .reg .b32 %temp; 
  mov.b64   {%r8, %temp}, %fd2;
  }
  {
  .reg .b32 %temp; 
  mov.b64   {%temp, %r9}, %fd2;
  }
  and.b32    %r10, %r9, 2147483647;
  setp.eq.s32   %p8, %r10, 2146435072;
  setp.eq.s32   %p9, %r8, 0;
  and.pred    %p10, %p8, %p9;
  xor.b32    %r11, %r9, %r1;
  setp.gt.s32   %p11, %r11, -1;
  and.pred    %p12, %p11, %p10;
  @%p12 bra   LBB9_6;
LBB9_4:
  ld.param.f32   %f1, [__xla_fp64_comparison_param_2];
  sub.f64   %fd5, %fd1, %fd2;
  abs.f64   %fd6, %fd5;
  max.f64   %fd7, %fd3, %fd4;
  add.f64   %fd8, %fd7, 0d3FF0000000000000;
  div.rn.f64   %fd9, %fd6, %fd8;
  cvt.f64.f32   %fd10, %f1;
  setp.leu.f64   %p13, %fd9, %fd10;
  abs.f64   %fd11, %fd9;
  setp.le.f64   %p14, %fd11, 0d7FF0000000000000;
  and.pred    %p15, %p13, %p14;
  @%p15 bra   LBB9_6;
  ld.param.u64   %rd6, [__xla_fp64_comparison_param_4];
  cvta.to.global.u64   %rd1, %rd6;
  atom.global.add.u32   %r12, [%rd1], 1;
LBB9_6:
  ret;
}
)";

template <typename ElementT>
using ComparisonKernelT =
    se::TypedKernel<se::DeviceMemory<ElementT>, se::DeviceMemory<ElementT>,
                    float, uint64, se::DeviceMemory<uint64>>;

// Compares two buffers on the GPU.
//
// Returns `true` if two buffers are equal, `false` otherwise.
template <typename ElementT>
static StatusOr<bool> DeviceCompare(se::Stream* stream,
                                    se::DeviceMemoryBase lhs,
                                    se::DeviceMemoryBase rhs,
                                    const Shape& buffer_shape,
                                    const HloModuleConfig& config,
                                    absl::string_view kernel_name) {
  se::StreamExecutor* executor = stream->parent();

  se::ScopedDeviceMemory<uint64> out_param =
      executor->AllocateOwnedScalar<uint64>();

  stream->ThenMemZero(out_param.ptr(), sizeof(uint64));
  if (lhs.size() != rhs.size()) {
    return InternalError("Mismatched buffer size: %d bytes vs. %d bytes",
                         lhs.size(), rhs.size());
  }

  se::DeviceMemory<ElementT> lhs_typed(lhs);
  se::DeviceMemory<ElementT> rhs_typed(rhs);
  uint64 buffer_size = lhs_typed.ElementCount();

  TF_ASSIGN_OR_RETURN(absl::Span<const uint8> compiled_ptx,
                      se::cuda::CompilePtxOrGetCached(
                          executor->device_ordinal(), buffer_compare_ptx,
                          PtxOptsFromConfig(config)));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<ComparisonKernelT<ElementT>> comparison_kernel,
      (executor->CreateTypedKernel<se::DeviceMemory<ElementT>,
                                   se::DeviceMemory<ElementT>, float, uint64,
                                   se::DeviceMemory<uint64>>(
          kernel_name, buffer_compare_ptx, compiled_ptx)));

  LaunchDimensions dim =
      CalculateLaunchDimensions(buffer_shape, executor->GetDeviceDescription());

  stream->ThenLaunch(se::ThreadDim(dim.threads_per_block()),
                     se::BlockDim(dim.block_count()), *comparison_kernel,
                     lhs_typed, rhs_typed, static_cast<float>(kTolerance),
                     buffer_size, out_param.cref());

  uint64 result = -1;
  CHECK_EQ(out_param->size(), sizeof(result));
  stream->ThenMemcpy(&result, *out_param, sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return result == 0;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
//
// Returns true if no differences were seen, false otherwise.
template <typename ElementType, typename ComparisonType>
StatusOr<bool> HostCompare(se::Stream* stream, se::DeviceMemoryBase lhs,
                           se::DeviceMemoryBase rhs) {
  int64 n = lhs.size() / sizeof(ElementType);
  std::vector<ElementType> host_lhs(n), host_rhs(n);
  stream->ThenMemcpy(host_lhs.data(), lhs, lhs.size());
  stream->ThenMemcpy(host_rhs.data(), rhs, rhs.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const auto canonicalize = [](ComparisonType a) -> ComparisonType {
    if (std::is_same<ElementType, Eigen::half>::value && a) {
      constexpr ComparisonType kMaxFp16Value = 65505.;
      if (std::isnan(a)) {
        return a;
      }
      return std::max(-kMaxFp16Value, std::min(a, kMaxFp16Value));
    }
    return a;
  };
  int differences_seen = 0;
  for (int64 i = 0; i < n && differences_seen < 10; i++) {
    auto original_lhs = static_cast<ComparisonType>(host_lhs[i]);
    auto original_rhs = static_cast<ComparisonType>(host_rhs[i]);
    ComparisonType lhs = canonicalize(original_lhs);
    ComparisonType rhs = canonicalize(original_rhs);
    if (std::isnan(lhs) && std::isnan(rhs)) {
      continue;
    }
    if (std::isinf(lhs) && std::isinf(rhs) && lhs == rhs) {
      continue;
    }
    if (std::isfinite(lhs) != std::isfinite(rhs) ||
        !(std::abs(lhs - rhs) / (std::max(std::abs(lhs), std::abs(rhs)) + 1) <
          kTolerance)) {
      differences_seen++;
      LOG(ERROR) << "Difference at " << i << ": " << original_lhs << " vs "
                 << original_rhs;
    }
  }
  return differences_seen == 0;
}

template <typename ElementT, typename ComparisonT>
static StatusOr<bool> CompareEqualParameterized(se::Stream* stream,
                                                se::DeviceMemoryBase lhs,
                                                se::DeviceMemoryBase rhs,
                                                const Shape& shape,
                                                const HloModuleConfig& config,
                                                absl::string_view kernel_name) {
  XLA_SCOPED_LOGGING_TIMER("BufferComparator::CompareEqual");
  TF_ASSIGN_OR_RETURN(
      bool result,
      DeviceCompare<ElementT>(stream, lhs, rhs, shape, config, kernel_name));

  if (result) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(bool host_return,
                      (HostCompare<ElementT, ComparisonT>(stream, lhs, rhs)));
  CHECK(host_return == result) << "Different comparison result on GPU vs host";

  return false;
}

StatusOr<bool> BufferComparator::CompareEqual(se::Stream* stream,
                                              se::DeviceMemoryBase lhs,
                                              se::DeviceMemoryBase rhs) const {
  switch (shape_.element_type()) {
    case xla::F16:
      return CompareEqualParameterized<Eigen::half, float>(
          stream, lhs, rhs, shape_, config_, "__xla_fp16_comparison");
    case xla::F32:
      return CompareEqualParameterized<float, float>(
          stream, lhs, rhs, shape_, config_, "__xla_fp32_comparison");
    case xla::F64:
      return CompareEqualParameterized<double, double>(
          stream, lhs, rhs, shape_, config_, "__xla_fp64_comparison");
    default:
      return Unimplemented("Unimplemented element type");
  }
}

BufferComparator::BufferComparator(const Shape& shape,
                                   const HloModuleConfig& config)
    : shape_(shape), config_(config) {
  // Normalize complex shapes: since we treat the passed array as a contiguous
  // storage it does not matter which dimension are we doubling.
  auto double_dim_size = [&]() {
    int64 prev_zero_dim_size = shape_.dimensions(0);
    shape_.set_dimensions(0, prev_zero_dim_size * 2);
  };

  if (shape_.element_type() == PrimitiveType::C64) {
    // C64 is just two F32s next to each other.
    shape_.set_element_type(PrimitiveType::F32);
    double_dim_size();
  } else if (shape_.element_type() == PrimitiveType::C128) {
    // C128 is just two F64s next to each other.
    shape_.set_element_type(PrimitiveType::F64);
    double_dim_size();
  }
}

}  // namespace gpu
}  // namespace xla
