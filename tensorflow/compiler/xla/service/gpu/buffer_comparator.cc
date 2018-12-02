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

#include <cmath>
#include "absl/strings/str_replace.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace gpu {

static constexpr float kTolerance = 0.1f;

static string GetCompHloText(size_t num_elements) {
  // Implements the textual format of the comparison routine, as it's more
  // readable.
  static constexpr char kF16CompHloText[] = R"(
HloModule CompareF16

MaxF32 {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %max = f32[] maximum(%lhs, %rhs)
}

Canonicalize (aparam: f16[SIZE]) -> f32[SIZE] {
  %min_constant = f32[] constant(-65505)
  %max_constant = f32[] constant(65505)
  %large_constant = f32[] constant(1048576)
  %min_values = f32[SIZE] broadcast(%min_constant), dimensions={}
  %max_values = f32[SIZE] broadcast(%max_constant), dimensions={}
  %large_values = f32[SIZE] broadcast(%large_constant), dimensions={}

  %a = f16[SIZE] parameter(0)
  %converted = f32[SIZE] convert(%a)
  %clamped = f32[SIZE] clamp(%min_values, %converted, %max_values)

  // Since the clamp() above already took care of infs, only NaNs will cause
  // is-finite() to return false.
  %is_finite = pred[SIZE] is-finite(%clamped)
  ROOT %result = f32[SIZE] select(%is_finite, %clamped, %large_values)
}

ENTRY MaxDifference {
  %one_constant = f32[] constant(1.0)
  %zero_constant = f32[] constant(0.0)

  %ones = f32[SIZE] broadcast(%one_constant), dimensions={}

  %lhs = f16[SIZE] parameter(0)
  %rhs = f16[SIZE] parameter(1)
  %lhs_canonical = f32[SIZE] call(%lhs), to_apply=Canonicalize
  %rhs_canonical = f32[SIZE] call(%rhs), to_apply=Canonicalize
  %sub = f32[SIZE] subtract(%lhs_canonical, %rhs_canonical)
  %sub_abs = f32[SIZE] abs(%sub)
  %lhs_abs = f32[SIZE] abs(%lhs_canonical)
  %rhs_abs = f32[SIZE] abs(%rhs_canonical)
  %max = f32[SIZE] maximum(%lhs_abs, %rhs_abs)
  %denominator = f32[SIZE] add(%max, %ones)
  %error = f32[SIZE] divide(%sub_abs, %denominator)
  ROOT %max_diff = f32[] reduce(%error, %zero_constant), dimensions={0}, to_apply=MaxF32
})";
  return absl::StrReplaceAll(kF16CompHloText,
                             {{"SIZE", absl::StrCat(num_elements)}});
}

StatusOr<F16BufferComparator> F16BufferComparator::Create(
    se::DeviceMemory<Eigen::half> ref_buffer, Compiler* compiler,
    DeviceMemoryAllocator* allocator, se::Stream* stream) {
  auto stream_exec = stream->parent();
  int64 num_elements = ref_buffer.ElementCount();

  // One may consider using hlo_runner to do all the compilation and execution.
  // However, as of the time hlo_runner doesn't support injection for Compiler*,
  // Stream*, or even the allocator. We may revisit this in the future if it
  // proves to be a maintenance burden.
  TF_ASSIGN_OR_RETURN(
      auto exec, ([&]() -> StatusOr<std::unique_ptr<Executable>> {
        HloModuleConfig config;
        DebugOptions debug_options;
        debug_options.set_xla_backend_optimization_level(2);
        config.set_debug_options(debug_options);
        TF_ASSIGN_OR_RETURN(
            auto module, ParseHloString(GetCompHloText(num_elements), config));
        TF_ASSIGN_OR_RETURN(
            module,
            compiler->RunHloPasses(std::move(module), stream_exec, nullptr));
        return compiler->RunBackend(std::move(module), stream_exec, nullptr);
      }()));

  TF_ASSIGN_OR_RETURN(
      auto shaped_buffer, ([&]() -> StatusOr<ScopedShapedBuffer> {
        auto device_ordinal = stream_exec->device_ordinal();
        TF_ASSIGN_OR_RETURN(
            auto owning_buffer,
            allocator->Allocate(device_ordinal, ref_buffer.size()));
        se::DeviceMemory<Eigen::half> buffer(
            owning_buffer.AsDeviceMemoryBase());
        stream->ThenMemcpy(&buffer, ref_buffer, ref_buffer.size());
        Shape shape = ShapeUtil::MakeShape(xla::F16, {num_elements});
        ScopedShapedBuffer ret(shape, shape, allocator, device_ordinal);
        ret.set_buffer(std::move(owning_buffer), {});
        return std::move(ret);
      }()));

  return F16BufferComparator(stream, allocator, std::move(exec),
                             std::move(shaped_buffer));
}

StatusOr<bool> F16BufferComparator::CompareEqualImpl(
    se::DeviceMemory<Eigen::half> test_buffer) {
  if (ref_buffer_.root_buffer().size() != test_buffer.size()) {
    return InternalError("Mismatched buffer size: %d vs %d",
                         ref_buffer_.root_buffer().size(), test_buffer.size());
  }

  int64 num_elements = test_buffer.ElementCount();

  TF_ASSIGN_OR_RETURN(
      auto result_buffer, ([&]() -> StatusOr<ScopedShapedBuffer> {
        auto stream_exec = stream_->parent();
        Shape shape = ShapeUtil::MakeShape(xla::F16, {num_elements});
        auto device_ordinal = stream_exec->device_ordinal();
        ShapedBuffer shaped_test_buffer(shape, shape, stream_exec->platform(),
                                        device_ordinal);
        shaped_test_buffer.set_buffer(test_buffer, {});
        ExecutableRunOptions run_options;
        run_options.set_device_ordinal(stream_exec->device_ordinal());
        run_options.set_stream(stream_);
        run_options.set_allocator(allocator_);
        ServiceExecutableRunOptions service_run_options(run_options);
        return exec_->ExecuteOnStream(
            &service_run_options, {&ref_buffer_, &shaped_test_buffer}, nullptr);
      }()));

  float result;
  CHECK(result_buffer.root_buffer().size() == sizeof(result));
  stream_->ThenMemcpy(&result, result_buffer.root_buffer(), sizeof(result));
  TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());
  return result < kTolerance;
}

StatusOr<bool> F16BufferComparator::CompareEqual(
    se::DeviceMemory<Eigen::half> test_buffer) {
  TF_ASSIGN_OR_RETURN(auto result, CompareEqualImpl(test_buffer));
  if (result) {
    return true;
  }
  // Host side code that does the same thing, but report some of the
  // differences as well.
  int64 n = test_buffer.ElementCount();
  std::vector<half> host_ref_buffer(n), host_test_buffer(n);
  stream_->ThenMemcpy(host_ref_buffer.data(), ref_buffer_.root_buffer(),
                      ref_buffer_.root_buffer().size());
  stream_->ThenMemcpy(host_test_buffer.data(), test_buffer, test_buffer.size());
  TF_RETURN_IF_ERROR(stream_->BlockHostUntilDone());

  const auto canonicalize = [](float a) -> float {
    constexpr float kBigNumer = 1048576.;
    constexpr float kMaxFp16Value = 65504.;
    if (std::isnan(a)) {
      return kBigNumer;
    }
    if (std::isinf(a)) {
      if (a < 0) {
        return -(kMaxFp16Value + 1);
      }
      return kMaxFp16Value + 1;
    }
    return a;
  };
  int differences_seen = 0;
  for (int64 i = 0; i < n && differences_seen < 10; i++) {
    float original_ref = static_cast<float>(host_ref_buffer[i]);
    float original_test = static_cast<float>(host_test_buffer[i]);
    float ref = canonicalize(original_ref);
    float test = canonicalize(original_test);
    if (!(std::abs(ref - test) / (std::max(std::abs(ref), std::abs(test)) + 1) <
          kTolerance)) {
      differences_seen++;
      LOG(ERROR) << "Difference at " << i << ": " << original_ref << " vs "
                 << original_test;
    }
  }

  return false;
}

}  // namespace gpu
}  // namespace xla
