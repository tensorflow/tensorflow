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
#include "tensorflow/compiler/xla/shape_util.h"
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

StatusOr<BufferComparator> BufferComparator::Create(
    const Shape& shape, se::StreamExecutor* stream_exec, Compiler* compiler) {
  if (shape.element_type() != xla::F16) {
    return Unimplemented("Unimplemented element type");
  }

  // One may consider using hlo_runner to do all the compilation and execution.
  // However, as of the time hlo_runner doesn't support injection for Compiler*,
  // or Stream*. We may revisit this in the future if it
  // proves to be a maintenance burden.
  TF_ASSIGN_OR_RETURN(
      auto exec, ([&]() -> StatusOr<std::unique_ptr<Executable>> {
        HloModuleConfig config;
        DebugOptions debug_options;
        debug_options.set_xla_backend_optimization_level(2);
        config.set_debug_options(debug_options);
        TF_ASSIGN_OR_RETURN(
            auto module,
            ParseHloString(GetCompHloText(ShapeUtil::ElementsIn(shape)),
                           config));
        TF_ASSIGN_OR_RETURN(
            module,
            compiler->RunHloPasses(std::move(module), stream_exec, nullptr));
        return compiler->RunBackend(std::move(module), stream_exec, nullptr);
      }()));

  return BufferComparator(shape, std::move(exec));
}

StatusOr<bool> BufferComparator::CompareEqualImpl(
    se::Stream* stream, DeviceMemoryAllocator* allocator,
    se::DeviceMemoryBase lhs, se::DeviceMemoryBase rhs) {
  if (lhs.size() != rhs.size()) {
    return InternalError("Mismatched buffer size: %d bytes vs %d bytes",
                         lhs.size(), rhs.size());
  }

  auto stream_exec = stream->parent();
  auto to_shaped_buffer =
      [stream_exec,
       this](se::DeviceMemoryBase buffer) -> StatusOr<ShapedBuffer> {
    auto device_ordinal = stream_exec->device_ordinal();
    ShapedBuffer shaped(shape_, shape_, stream_exec->platform(),
                        device_ordinal);
    shaped.set_buffer(buffer, {});
    return std::move(shaped);
  };

  TF_ASSIGN_OR_RETURN(auto shaped_lhs, to_shaped_buffer(lhs));
  TF_ASSIGN_OR_RETURN(auto shaped_rhs, to_shaped_buffer(rhs));

  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_exec->device_ordinal());
  run_options.set_stream(stream);
  run_options.set_allocator(allocator);
  ServiceExecutableRunOptions service_run_options(run_options);

  const ShapedBuffer* arg_buffers[] = {&shaped_lhs, &shaped_rhs};
  TF_ASSIGN_OR_RETURN(auto result_buffer,
                      comparator_exec_->ExecuteOnStream(&service_run_options,
                                                        arg_buffers, nullptr));

  float result;
  CHECK(result_buffer.root_buffer().size() == sizeof(result));
  stream->ThenMemcpy(&result, result_buffer.root_buffer(), sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return result < kTolerance;
}

StatusOr<bool> BufferComparator::CompareEqual(se::Stream* stream,
                                              DeviceMemoryAllocator* allocator,
                                              se::DeviceMemoryBase lhs,
                                              se::DeviceMemoryBase rhs) {
  TF_ASSIGN_OR_RETURN(auto result,
                      CompareEqualImpl(stream, allocator, lhs, rhs));

  if (result) {
    return true;
  }

  // Host side code that does the same thing, but reports some of the
  // differences as well.
  int64 n = ShapeUtil::ElementsIn(shape_);
  std::vector<half> host_lhs(n), host_rhs(n);
  stream->ThenMemcpy(host_lhs.data(), lhs, lhs.size());
  stream->ThenMemcpy(host_rhs.data(), rhs, rhs.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const auto canonicalize = [](float a) -> float {
    constexpr float kBigNumber = 1048576.;
    constexpr float kMaxFp16Value = 65504.;
    if (std::isnan(a)) {
      return kBigNumber;
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
    float original_lhs = static_cast<float>(host_lhs[i]);
    float original_rhs = static_cast<float>(host_rhs[i]);
    float lhs = canonicalize(original_lhs);
    float rhs = canonicalize(original_rhs);
    if (!(std::abs(lhs - rhs) / (std::max(std::abs(lhs), std::abs(rhs)) + 1) <
          kTolerance)) {
      differences_seen++;
      LOG(ERROR) << "Difference at " << i << ": " << original_lhs << " vs "
                 << original_rhs;
    }
  }

  return false;
}

}  // namespace gpu
}  // namespace xla
