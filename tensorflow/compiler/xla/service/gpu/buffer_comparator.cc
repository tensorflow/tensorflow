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

static constexpr double kTolerance = 0.1f;

static StatusOr<string> GetCompHloText(const Shape& shape) {
  // Implements the textual format of the comparison routine, as it's more
  // readable.
  //
  // This text template takes three substitution parameters:
  // ${ORIG_TYPE}: buffer element type.
  // ${CMP_TYPE}: intermediate element type for calculating numeric differences.
  // ${SIZE}: number of elements.
  // ${CLAMP_TO}: Clamp the value to [-$CLAMP_TO, $CLAMP_TO].
  static constexpr char kCompHloText[] = R"(
HloModule Compare_${ORIG_TYPE}_${CMP_TYPE}_${SIZE}_${CLAMP_TO}

Max {
  %lhs = ${CMP_TYPE}[] parameter(0)
  %rhs = ${CMP_TYPE}[] parameter(1)
  ROOT %max = ${CMP_TYPE}[] maximum(%lhs, %rhs)
}

Canonicalize (aparam: ${ORIG_TYPE}[${SIZE}]) -> ${CMP_TYPE}[${SIZE}] {
  %min_constant = ${CMP_TYPE}[] constant(-${CLAMP_TO})
  %max_constant = ${CMP_TYPE}[] constant(${CLAMP_TO})
  %min_values = ${CMP_TYPE}[${SIZE}] broadcast(%min_constant), dimensions={}
  %max_values = ${CMP_TYPE}[${SIZE}] broadcast(%max_constant), dimensions={}

  %a = ${ORIG_TYPE}[${SIZE}] parameter(0)
  %converted = ${CMP_TYPE}[${SIZE}] convert(%a)
  ROOT %clamped = ${CMP_TYPE}[${SIZE}] clamp(%min_values, %converted, %max_values)
}

// RelError(x, y) = abs(x - y) / (max(abs(x), abs(y)) + 1)
// x and y must be finite.
RelError (aparam: ${CMP_TYPE}[${SIZE}], bparam: ${CMP_TYPE}[${SIZE}]) -> ${CMP_TYPE}[${SIZE}] {
  %lhs = ${CMP_TYPE}[${SIZE}] parameter(0)
  %rhs = ${CMP_TYPE}[${SIZE}] parameter(1)
  %one_constant = ${CMP_TYPE}[] constant(1.0)
  %ones = ${CMP_TYPE}[${SIZE}] broadcast(%one_constant), dimensions={}

  %sub = ${CMP_TYPE}[${SIZE}] subtract(%lhs, %rhs)
  %sub_abs = ${CMP_TYPE}[${SIZE}] abs(%sub)
  %lhs_abs = ${CMP_TYPE}[${SIZE}] abs(%lhs)
  %rhs_abs = ${CMP_TYPE}[${SIZE}] abs(%rhs)
  %max = ${CMP_TYPE}[${SIZE}] maximum(%lhs_abs, %rhs_abs)
  %denominator = ${CMP_TYPE}[${SIZE}] add(%max, %ones)
  ROOT %error = ${CMP_TYPE}[${SIZE}] divide(%sub_abs, %denominator)
}

// Here is the chain-style definition of this function:
//   Error(NaN, NaN) = 0
//   Error(Inf, Inf) = 0
//   Error(-Inf, -Inf) = 0
//   Error(NonFinite, x) = Inf
//   Error(x, NonFinite) = Inf
//   Error(x, y) = RelError(x, y)
// , where the early matched pattern takes precedence.
//
// To implement this, we start from the bottom, and keep using select to
// overwrite previously picked values. The last value produced by a matched
// pattern is the final value.
Error (aparam: ${CMP_TYPE}[${SIZE}], bparam: ${CMP_TYPE}[${SIZE}]) -> ${CMP_TYPE}[${SIZE}] {
  %lhs = ${CMP_TYPE}[${SIZE}] parameter(0)
  %rhs = ${CMP_TYPE}[${SIZE}] parameter(1)
  %zero_constant = ${CMP_TYPE}[] constant(0.0)
  %inf_constant = ${CMP_TYPE}[] constant(inf)
  %zeros = ${CMP_TYPE}[${SIZE}] broadcast(%zero_constant), dimensions={}
  %infs = ${CMP_TYPE}[${SIZE}] broadcast(%inf_constant), dimensions={}

  %lhs_is_finite = pred[${SIZE}] is-finite(%lhs)
  %lhs_is_not_finite = pred[${SIZE}] not(%lhs_is_finite)
  %lhs_is_not_nan = pred[${SIZE}] compare(%lhs, %lhs), direction=EQ
  %lhs_is_nan = pred[${SIZE}] not(%lhs_is_not_nan)
  %lhs_is_inf = pred[${SIZE}] and(%lhs_is_not_finite, %lhs_is_not_nan)
  %lhs_is_non_neg = pred[${SIZE}] compare(%lhs, %zeros), direction=GE

  %rhs_is_finite = pred[${SIZE}] is-finite(%rhs)
  %rhs_is_not_finite = pred[${SIZE}] not(%rhs_is_finite)
  %rhs_is_not_nan = pred[${SIZE}] compare(%rhs, %rhs), direction=EQ
  %rhs_is_nan = pred[${SIZE}] not(%rhs_is_not_nan)
  %rhs_is_inf = pred[${SIZE}] and(%rhs_is_not_finite, %rhs_is_not_nan)
  %rhs_is_non_neg = pred[${SIZE}] compare(%rhs, %zeros), direction=GE

  %both_same_sign = pred[${SIZE}] and(%lhs_is_non_neg, %rhs_is_non_neg)
  %both_inf = pred[${SIZE}] and(%lhs_is_inf, %rhs_is_inf)
  %both_same_sign_inf = pred[${SIZE}] and(%both_same_sign, %both_inf)
  %both_nan = pred[${SIZE}] and(%lhs_is_nan, %rhs_is_nan)

  // Reverse-order selections

  // Error(x, y) = RelError(x, y)
  %rel_error = ${CMP_TYPE}[${SIZE}] call(%lhs, %rhs), to_apply=RelError
  // Error(x, NonFinite) = Inf
  %after_x_non_finite = ${CMP_TYPE}[${SIZE}] select(%rhs_is_not_finite, %infs, %rel_error)
  // Error(NonFinite, x) = Inf
  %after_non_finite_x = ${CMP_TYPE}[${SIZE}] select(%lhs_is_not_finite, %infs, %after_x_non_finite)
  // Error(-Inf, -Inf) = 0
  // Error(Inf, Inf) = 0
  %after_both_same_sign_inf = ${CMP_TYPE}[${SIZE}] select(%both_same_sign_inf, %zeros, %after_non_finite_x)
  // Error(NaN, NaN) = 0
  ROOT %after_both_nan = ${CMP_TYPE}[${SIZE}] select(%both_nan, %zeros, %after_both_same_sign_inf)
}

ENTRY MaxDifference {
  %zero_constant = ${CMP_TYPE}[] constant(0.0)

  %lhs = ${ORIG_TYPE}[${SIZE}] parameter(0)
  %rhs = ${ORIG_TYPE}[${SIZE}] parameter(1)
  %lhs_canonical = ${CMP_TYPE}[${SIZE}] call(%lhs), to_apply=Canonicalize
  %rhs_canonical = ${CMP_TYPE}[${SIZE}] call(%rhs), to_apply=Canonicalize
  %error = ${CMP_TYPE}[${SIZE}] call(%lhs_canonical, %rhs_canonical), to_apply=Error
  %max_diff = ${CMP_TYPE}[] reduce(%error, %zero_constant), dimensions={0}, to_apply=Max
  ROOT %converted_max_diff = f64[] convert(%max_diff)
})";

  absl::string_view orig_type;
  absl::string_view cmp_type;
  string clamp_to;

  switch (shape.element_type()) {
    case xla::F16:
      orig_type = "f16";
      cmp_type = "f32";
      // Clamp fp16s to 65505, since they actually overflow a lot in practice.
      // This way, +infs and values like 65504 are considered be within
      // tolerance.
      clamp_to = "65505";
      break;
    case xla::F32:
      orig_type = "f32";
      cmp_type = "f32";
      clamp_to = "inf";
      break;
    case xla::F64:
      orig_type = "f64";
      cmp_type = "f64";
      clamp_to = "inf";
      break;
    default:
      return Unimplemented("Unimplemented element type");
  }

  string size_str = absl::StrCat(ShapeUtil::ElementsIn(shape));
  return absl::StrReplaceAll(kCompHloText, {
                                               {"${ORIG_TYPE}", orig_type},
                                               {"${CMP_TYPE}", cmp_type},
                                               {"${SIZE}", size_str},
                                               {"${CLAMP_TO}", clamp_to},
                                           });
}

StatusOr<BufferComparator> BufferComparator::Create(
    const Shape& shape, se::StreamExecutor* stream_exec, Compiler* compiler) {
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
        TF_ASSIGN_OR_RETURN(string hlo_text, GetCompHloText(shape));
        TF_ASSIGN_OR_RETURN(auto module, ParseHloString(hlo_text, config));
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

  double result;
  CHECK(result_buffer.root_buffer().size() == sizeof(result));
  stream->ThenMemcpy(&result, result_buffer.root_buffer(), sizeof(result));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return result < kTolerance;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
template <typename ElementType, typename ComparisonType>
Status HostCompare(se::Stream* stream, se::DeviceMemoryBase lhs,
                   se::DeviceMemoryBase rhs) {
  int64 n = lhs.size() / sizeof(ElementType);
  std::vector<ElementType> host_lhs(n), host_rhs(n);
  stream->ThenMemcpy(host_lhs.data(), lhs, lhs.size());
  stream->ThenMemcpy(host_rhs.data(), rhs, rhs.size());
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  const auto canonicalize = [](ComparisonType a) -> ComparisonType {
    if (std::is_same<ElementType, Eigen::half>::value && a) {
      constexpr float kMaxFp16Value = 65504.;
      if (std::isnan(a)) {
        return a;
      }
      if (a < 0) {
        return -(kMaxFp16Value + 1);
      }
      return kMaxFp16Value + 1;
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
  return Status::OK();
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

  switch (shape_.element_type()) {
    case xla::F16:
      TF_RETURN_IF_ERROR(HostCompare<Eigen::half, float>(stream, lhs, rhs));
      break;
    case xla::F32:
      TF_RETURN_IF_ERROR(HostCompare<float, float>(stream, lhs, rhs));
      break;
    case xla::F64:
      TF_RETURN_IF_ERROR(HostCompare<double, double>(stream, lhs, rhs));
      break;
    default:
      return Unimplemented("Unimplemented element type");
  }

  return false;
}

}  // namespace gpu
}  // namespace xla
