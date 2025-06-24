/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/aot/thunk_proto_execution_deserializer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/layout_util.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace tensorflow {
namespace tfcompile {

namespace {

std::string GetBufferAllocationString(
    const xla::buffer_assignment::BufferAllocationSliceProto& slice) {
  return absl::StrCat("reinterpret_cast<std::byte*>(buffer_table[",
                      slice.buffer_allocation_index(), "]) + ", slice.offset());
}

}  // namespace

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetThunkSpecificRunImpl(
    const xla::cpu::CompilationResultProto& proto) && {
  return ThunkSpecificRunImplFromThunkSequence(proto.thunk_sequence());
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::ThunkSpecificRunImplFromThunkSequence(
    const xla::cpu::ThunkSequenceProto& thunk_sequence_proto) {
  std::vector<std::string> thunk_run_impls;
  thunk_run_impls.reserve(thunk_sequence_proto.thunks_size());

  for (const auto& thunk : thunk_sequence_proto.thunks()) {
    switch (thunk.impl_case()) {
      case xla::cpu::ThunkProto::kKernelThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetKernelThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kDotThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetDotThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kConditionalThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetConditionalThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kWhileThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetWhileThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kConvolutionThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetConvolutionFusionThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kRngGetAndUpdateStateThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetRngGetAndUpdateStateThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kCallThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetCallThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kCopyThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetCopyThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kSortThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetSortThunkRunImpl(thunk));
        break;
      }
      case xla::cpu::ThunkProto::kTopKThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetTopKThunkRunImpl(thunk));
        break;
      }
      default: {
        return xla::Internal("Unsupported thunk type: %s.", thunk.kind());
      }
    }
  }

  return absl::StrJoin(thunk_run_impls, "\n");
}

absl::StatusOr<std::string> ThunkProtoExecutionDeserializer::GetMatmulFunction(
    xla::PrimitiveType xla_type, bool is_single_threaded) {
  switch (xla_type) {
    case xla::F16:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulF16"
                 : "__xla_cpu_runtime_EigenMatMulF16";
    case xla::F32:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulF32"
                 : "__xla_cpu_runtime_EigenMatMulF32";
    case xla::F64:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulF64"
                 : "__xla_cpu_runtime_EigenMatMulF64";
    case xla::C64:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulC64"
                 : "__xla_cpu_runtime_EigenMatMulC64";
    case xla::C128:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulC128"
                 : "__xla_cpu_runtime_EigenMatMulC128";
    case xla::S32:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedMatMulS32"
                 : "__xla_cpu_runtime_EigenMatMulS32";
    default:
      return xla::Internal("Unsupported xla type: %d", xla_type);
  }
}

absl::StatusOr<std::string> ThunkProtoExecutionDeserializer::GetDotThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_dot_thunk()) {
    return xla::Internal(
        "Dot thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::DotThunkProto& dot_thunk = thunk.dot_thunk();

  absl::string_view dot_thunk_invocation_format = xla_cpu_multi_thread_eigen_
                                                      ? R"(
     // Dot Thunk
     {
        for (int64_t i = 0; i < {{BATCH_SIZE}}; ++i) {
          if (run_options->intra_op_thread_pool() != nullptr) {
            {{MATMUL_FUNCTION}}(
              run_options,
              {{OUTPUT_PTR}} + {{OUTPUT_STRIDE}} * i,
              {{LHS_PTR}} + {{LHS_STRIDE}} * i,
              {{RHS_PTR}} + {{RHS_STRIDE}} * i,
              {{M}}, {{N}}, {{K}}, {{TRANSPOSE_LHS}}, {{TRANSPOSE_RHS}});
          } else {
            {{SINGLE_THREADED_MATMUL_FUNCTION}}(
                nullptr, 
                {{OUTPUT_PTR}} + {{OUTPUT_STRIDE}} * i,
                {{LHS_PTR}} + {{LHS_STRIDE}} * i,
                {{RHS_PTR}} + {{RHS_STRIDE}} * i,
                {{M}}, {{N}}, {{K}}, {{TRANSPOSE_LHS}}, {{TRANSPOSE_RHS}});
          }
        }
     }
     )"
                                                      :
                                                      R"(
      // Dot Thunk
      {
         for (int64_t i = 0; i < {{BATCH_SIZE}}; ++i) {
          {{SINGLE_THREADED_MATMUL_FUNCTION}}(
                nullptr,
                {{OUTPUT_PTR}} + {{OUTPUT_STRIDE}} * i,
                {{LHS_PTR}} + {{LHS_STRIDE}} * i,
                {{RHS_PTR}} + {{RHS_STRIDE}} * i,
                {{M}}, {{N}}, {{K}}, {{TRANSPOSE_LHS}}, {{TRANSPOSE_RHS}});
         }
      }
      )";

  if (!(dot_thunk.lhs_buffer_shape().shape().element_type() ==
            dot_thunk.rhs_buffer_shape().shape().element_type() &&
        dot_thunk.rhs_buffer_shape().shape().element_type() ==
            dot_thunk.out_buffer_shape().shape().element_type())) {
    return xla::Internal(
        "Dot thunk has mismatched types between lhs, rhs, and out buffers.");
  }

  TF_ASSIGN_OR_RETURN(
      std::string matmul_function,
      GetMatmulFunction(dot_thunk.lhs_buffer_shape().shape().element_type(),
                        /*is_single_threaded=*/false));

  TF_ASSIGN_OR_RETURN(
      std::string single_threaded_matmul_function,
      GetMatmulFunction(dot_thunk.lhs_buffer_shape().shape().element_type(),
                        /*is_single_threaded=*/true));

  TF_ASSIGN_OR_RETURN(std::string data_type,
                      CppDataTypeFromXlaType(
                          dot_thunk.lhs_buffer_shape().shape().element_type()));

  std::string output_ptr = absl::StrCat(
      "reinterpret_cast<", data_type, "*>(",
      GetBufferAllocationString(dot_thunk.out_buffer_shape().slice()), ")");
  std::string lhs_ptr = absl::StrCat(
      "reinterpret_cast<", data_type, "*>(",
      GetBufferAllocationString(dot_thunk.lhs_buffer_shape().slice()), ")");
  std::string rhs_ptr = absl::StrCat(
      "reinterpret_cast<", data_type, "*>(",
      GetBufferAllocationString(dot_thunk.rhs_buffer_shape().slice()), ")");

  TF_ASSIGN_OR_RETURN(
      auto lhs_shape,
      xla::Shape::FromProto(dot_thunk.lhs_buffer_shape().shape()));
  TF_ASSIGN_OR_RETURN(
      auto rhs_shape,
      xla::Shape::FromProto(dot_thunk.rhs_buffer_shape().shape()));
  TF_ASSIGN_OR_RETURN(
      auto out_shape,
      xla::Shape::FromProto(dot_thunk.out_buffer_shape().shape()));

  TF_ASSIGN_OR_RETURN(xla::cpu::DotShape dot_shape,
                      xla::cpu::GetDotShape(dot_thunk.dot_dimensions(),
                                            lhs_shape, rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(
      xla::cpu::DotCanonicalDims dot_canonical_dims,
      GetDotCanonicalDims(dot_thunk.dot_dimensions(), dot_shape));

  size_t m = dot_canonical_dims.m;
  size_t k = dot_canonical_dims.k;
  size_t n = dot_canonical_dims.n;

  // Decide if a transpose is required based on an XOR of the canonical and
  // column major flags.
  bool transpose_lhs =
      (dot_canonical_dims.lhs_canonical != dot_canonical_dims.lhs_column_major);
  bool transpose_rhs =
      (dot_canonical_dims.rhs_canonical != dot_canonical_dims.rhs_column_major);

  if (!dot_canonical_dims.output_column_major) {
    std::swap(m, n);
    std::swap(lhs_ptr, rhs_ptr);
    std::swap(transpose_lhs, transpose_rhs);
    transpose_lhs = !transpose_lhs;
    transpose_rhs = !transpose_rhs;
  }

  // NOTE: Don't use byte width because the buffer pointers will already be cast
  // to the correct type. Reference xla/backends/cpu/runtime/dot_thunk.cc.
  int64_t lhs_stride = m * k;
  int64_t rhs_stride = k * n;
  int64_t out_stride = m * n;

  std::vector<std::pair<std::string, std::string>> rewrites = {
      {"{{SINGLE_THREADED_MATMUL_FUNCTION}}", single_threaded_matmul_function},
      {"{{OUTPUT_PTR}}", output_ptr},
      {"{{OUTPUT_STRIDE}}", absl::StrCat(out_stride)},
      {"{{LHS_PTR}}", lhs_ptr},
      {"{{LHS_STRIDE}}", absl::StrCat(lhs_stride)},
      {"{{RHS_PTR}}", rhs_ptr},
      {"{{RHS_STRIDE}}", absl::StrCat(rhs_stride)},
      {"{{M}}", absl::StrCat(m)},
      {"{{N}}", absl::StrCat(n)},
      {"{{K}}", absl::StrCat(k)},
      {"{{TRANSPOSE_LHS}}", transpose_lhs ? "true" : "false"},
      {"{{TRANSPOSE_RHS}}", transpose_rhs ? "true" : "false"},
      {"{{BATCH_SIZE}}", absl::StrCat(dot_shape.batch_size)}};

  if (xla_cpu_multi_thread_eigen_) {
    rewrites.push_back({"{{MATMUL_FUNCTION}}", matmul_function});
  }

  return absl::StrReplaceAll(dot_thunk_invocation_format, rewrites);
};

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetConvolutionFunction(
    xla::PrimitiveType xla_type, bool is_single_threaded) {
  switch (xla_type) {
    case xla::F16:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedConv2DF16"
                 : "__xla_cpu_runtime_EigenConv2DF16";
    case xla::F32:
      return is_single_threaded
                 ? "__xla_cpu_runtime_EigenSingleThreadedConv2DF32"
                 : "__xla_cpu_runtime_EigenConv2DF32";
    default:
      return xla::Internal("Unsupported xla type: %d", xla_type);
  }
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetConvolution2DRunImpl(
    const xla::cpu::ConvolutionThunkProto& convolution_thunk,
    const xla::cpu::ConvolutionCanonicalDims& canonical_dims) {
  TF_ASSIGN_OR_RETURN(
      std::string data_type,
      CppDataTypeFromXlaType(
          convolution_thunk.input_buffer_shape().shape().element_type()));

  std::string output_ptr =
      absl::StrCat("reinterpret_cast<", data_type, "*>(",
                   GetBufferAllocationString(
                       convolution_thunk.output_buffer_shape().slice()),
                   ")");
  std::string lhs_ptr = absl::StrCat(
      "reinterpret_cast<", data_type, "*>(",
      GetBufferAllocationString(convolution_thunk.input_buffer_shape().slice()),
      ")");
  std::string rhs_ptr =
      absl::StrCat("reinterpret_cast<", data_type, "*>(",
                   GetBufferAllocationString(
                       convolution_thunk.kernel_buffer_shape().slice()),
                   ")");

  TF_ASSIGN_OR_RETURN(
      std::string convolution_function,
      GetConvolutionFunction(
          convolution_thunk.input_buffer_shape().shape().element_type(),
          /*is_single_threaded=*/false));

  TF_ASSIGN_OR_RETURN(
      std::string single_threaded_convolution_function,
      GetConvolutionFunction(
          convolution_thunk.input_buffer_shape().shape().element_type(),
          /*is_single_threaded=*/true));

  absl::string_view convolution_thunk_invocation_format =
      xla_cpu_multi_thread_eigen_ ? R"(
     // Convolution Thunk
     {
         if (run_options->intra_op_thread_pool() != nullptr) {
           {{CONVOLUTION_FUNCTION}}(
             run_options,
             {{OUTPUT_PTR}}, {{LHS_PTR}}, {{RHS_PTR}}, {{INPUT_BATCH}},
             {{INPUT_ROWS}}, {{INPUT_COLS}}, {{INPUT_CHANNELS}}, {{KERNEL_ROWS}},
             {{KERNEL_COLS}}, {{KERNEL_CHANNELS}}, {{KERNEL_FILTERS}},
             {{OUTPUT_ROWS}}, {{OUTPUT_COLS}}, {{ROW_STRIDE}}, {{COL_STRIDE}},
             {{PADDING_TOP}}, {{PADDING_BOTTOM}}, {{PADDING_LEFT}},
             {{PADDING_RIGHT}}, {{LHS_ROW_DILATION}}, {{LHS_COL_DILATION}},
             {{RHS_ROW_DILATION}}, {{RHS_COL_DILATION}}, {{FEATURE_GROUP_COUNT}}
           );
         } else {
           {{SINGLE_THREADED_CONVOLUTION_FUNCTION}}(
             nullptr,
             {{OUTPUT_PTR}}, {{LHS_PTR}}, {{RHS_PTR}}, {{INPUT_BATCH}},
             {{INPUT_ROWS}}, {{INPUT_COLS}}, {{INPUT_CHANNELS}}, {{KERNEL_ROWS}},
             {{KERNEL_COLS}}, {{KERNEL_CHANNELS}}, {{KERNEL_FILTERS}},
             {{OUTPUT_ROWS}}, {{OUTPUT_COLS}}, {{ROW_STRIDE}}, {{COL_STRIDE}},
             {{PADDING_TOP}}, {{PADDING_BOTTOM}}, {{PADDING_LEFT}},
             {{PADDING_RIGHT}}, {{LHS_ROW_DILATION}}, {{LHS_COL_DILATION}},
             {{RHS_ROW_DILATION}}, {{RHS_COL_DILATION}}, {{FEATURE_GROUP_COUNT}}
           );
         }
     })"
                                  :
                                  R"(
      // Convolution Thunk
      {
        {{SINGLE_THREADED_CONVOLUTION_FUNCTION}}(
              nullptr,
              {{OUTPUT_PTR}}, {{LHS_PTR}}, {{RHS_PTR}}, {{INPUT_BATCH}},
              {{INPUT_ROWS}}, {{INPUT_COLS}}, {{INPUT_CHANNELS}}, {{KERNEL_ROWS}},
              {{KERNEL_COLS}}, {{KERNEL_CHANNELS}}, {{KERNEL_FILTERS}},
              {{OUTPUT_ROWS}}, {{OUTPUT_COLS}}, {{ROW_STRIDE}}, {{COL_STRIDE}},
              {{PADDING_TOP}}, {{PADDING_BOTTOM}}, {{PADDING_LEFT}},
              {{PADDING_RIGHT}}, {{LHS_ROW_DILATION}}, {{LHS_COL_DILATION}},
              {{RHS_ROW_DILATION}}, {{RHS_COL_DILATION}}, {{FEATURE_GROUP_COUNT}}
            );
      }
      )";

  std::vector<std::pair<std::string, std::string>> rewrites = {
      {"{{SINGLE_THREADED_CONVOLUTION_FUNCTION}}",
       single_threaded_convolution_function},
      {"{{OUTPUT_PTR}}", output_ptr},
      {"{{LHS_PTR}}", lhs_ptr},
      {"{{RHS_PTR}}", rhs_ptr},
      {"{{INPUT_BATCH}}", absl::StrCat(canonical_dims.input_batch)},
      {"{{INPUT_ROWS}}", absl::StrCat(canonical_dims.input_dims.x)},
      {"{{INPUT_COLS}}", absl::StrCat(canonical_dims.input_dims.y)},
      {"{{INPUT_CHANNELS}}", absl::StrCat(canonical_dims.input_channels)},
      {"{{KERNEL_ROWS}}", absl::StrCat(canonical_dims.kernel_dims.x)},
      {"{{KERNEL_COLS}}", absl::StrCat(canonical_dims.kernel_dims.y)},
      {"{{KERNEL_CHANNELS}}", absl::StrCat(canonical_dims.kernel_channels)},
      {"{{KERNEL_FILTERS}}", absl::StrCat(canonical_dims.kernel_filters)},
      {"{{OUTPUT_ROWS}}", absl::StrCat(canonical_dims.output_dims.x)},
      {"{{OUTPUT_COLS}}", absl::StrCat(canonical_dims.output_dims.y)},
      {"{{ROW_STRIDE}}", absl::StrCat(canonical_dims.strides.x)},
      {"{{COL_STRIDE}}", absl::StrCat(canonical_dims.strides.y)},
      {"{{PADDING_TOP}}", absl::StrCat(canonical_dims.padding_before.x)},
      {"{{PADDING_BOTTOM}}", absl::StrCat(canonical_dims.padding_after.x)},
      {"{{PADDING_LEFT}}", absl::StrCat(canonical_dims.padding_before.y)},
      {"{{PADDING_RIGHT}}", absl::StrCat(canonical_dims.padding_after.y)},
      {"{{LHS_ROW_DILATION}}", absl::StrCat(canonical_dims.base_dilation.x)},
      {"{{LHS_COL_DILATION}}", absl::StrCat(canonical_dims.base_dilation.y)},
      {"{{RHS_ROW_DILATION}}", absl::StrCat(canonical_dims.window_dilation.x)},
      {"{{RHS_COL_DILATION}}", absl::StrCat(canonical_dims.window_dilation.y)},
      {"{{FEATURE_GROUP_COUNT}}",
       absl::StrCat(canonical_dims.feature_group_count)}};

  if (xla_cpu_multi_thread_eigen_) {
    rewrites.push_back({"{{CONVOLUTION_FUNCTION}}", convolution_function});
  }

  return absl::StrReplaceAll(convolution_thunk_invocation_format, rewrites);
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetConvolutionFusionThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_convolution_thunk()) {
    return xla::Internal(
        "Convolution thunk was expected when getting thunk run "
        "implementation.");
  }
  const xla::cpu::ConvolutionThunkProto& convolution_thunk =
      thunk.convolution_thunk();

  TF_ASSIGN_OR_RETURN(
      xla::Shape input_buffer_shape,
      xla::Shape::FromProto(convolution_thunk.input_buffer_shape().shape()));
  TF_ASSIGN_OR_RETURN(
      xla::Shape kernel_buffer_shape,
      xla::Shape::FromProto(convolution_thunk.kernel_buffer_shape().shape()));
  TF_ASSIGN_OR_RETURN(
      xla::Shape output_buffer_shape,
      xla::Shape::FromProto(convolution_thunk.output_buffer_shape().shape()));

  // NOTE(basioli): Slices are not needed here, we only use this class to
  // invoke GetConvolutionCanonicalDims.
  xla::cpu::ConvolutionSlices slices{
      /*input_buffer=*/{},
      /*input_shape=*/input_buffer_shape,
      /*kernel_buffer=*/{},
      /*kernel_shape=*/kernel_buffer_shape,
      /*output_buffer=*/{},
      /*output_shape=*/output_buffer_shape,
  };

  TF_ASSIGN_OR_RETURN(
      xla::cpu::ConvolutionCanonicalDims canonical_dims,
      xla::cpu::GetConvolutionCanonicalDims(
          slices, convolution_thunk.dimension_numbers(),
          convolution_thunk.window(), convolution_thunk.feature_group_count()));

  if (canonical_dims.convolution_rank() == 2) {
    return GetConvolution2DRunImpl(convolution_thunk, canonical_dims);
  } else {
    return xla::Internal("3D convolution is not implemented.");
  }
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetRngGetAndUpdateStateThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_rng_get_and_update_state_thunk()) {
    return xla::Internal(
        "RngGetAndUpdateState thunk was expected when getting thunk run "
        "implementation.");
  }
  const xla::cpu::RngGetAndUpdateStateThunkProto& rng_thunk =
      thunk.rng_get_and_update_state_thunk();
  absl::string_view rng_thunk_invocation_format = R"(
     // Rng Thunk
     {
         rng_states[{{RNG_STATE_INDEX}}]->GetAndUpdateState({{RNG_STATE_PTR}});
     })";

  if (rng_thunk.state_buffer().size() != sizeof(absl::int128)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Rng state buffer size: ", rng_thunk.state_buffer().size(),
                     " is not equal to the size of an absl::int128: ",
                     sizeof(absl::int128)));
  }

  return absl::StrReplaceAll(
      rng_thunk_invocation_format,
      {{"{{RNG_STATE_INDEX}}", absl::StrCat(rng_state_index_++)},
       {"{{RNG_STATE_PTR}}",
        absl::StrCat("reinterpret_cast<uint64_t*>(",
                     GetBufferAllocationString(rng_thunk.state_buffer()),
                     ")")}});
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetCallThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_call_thunk()) {
    return xla::Internal(
        "Calls thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::CallThunkProto& call_thunk = thunk.call_thunk();
  absl::string_view call_thunk_invocation_format = R"(
     // Call Thunk
     {
         {{CALL_THUNK_IMPL}}
     })";

  TF_ASSIGN_OR_RETURN(
      std::string call_thunk_impl,
      ThunkSpecificRunImplFromThunkSequence(call_thunk.called_sequence()));

  return absl::StrReplaceAll(call_thunk_invocation_format,
                             {{"{{CALL_THUNK_IMPL}}", call_thunk_impl}});
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetCopyThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  // IMPORTANT(basioli): tfcompiled models should always emit llvm kernels for
  // copy thunks. Here we emit just a memcpy. This is done exclusively for copy
  // thunks that get created by the sort thunk.
  if (!thunk.has_copy_thunk()) {
    return xla::Internal(
        "Copy thunk was expected when getting thunk run implementation.");
  }

  const xla::cpu::CopyThunkProto& copy_thunk = thunk.copy_thunk();
  TF_ASSIGN_OR_RETURN(
      auto input_shape,
      xla::Shape::FromProto(copy_thunk.src_buffer_shape().shape()));
  TF_ASSIGN_OR_RETURN(
      auto output_shape,
      xla::Shape::FromProto(copy_thunk.dst_buffer_shape().shape()));

  if (input_shape != output_shape) {
    return xla::Internal(
        "Copy thunk has input shape %s and output shape %s that are not the "
        "same.",
        input_shape.ToString(true), output_shape.ToString(true));
  }

  absl::string_view copy_thunk_invocation_format = R"(
     // Copy Thunk
     {
         std::memcpy({{OUTPUT_PTR}}, {{INPUT_PTR}}, {{SIZE}});
     })";

  return absl::StrReplaceAll(
      copy_thunk_invocation_format,
      {{"{{OUTPUT_PTR}}",
        absl::StrCat(
            "reinterpret_cast<char*>(",
            GetBufferAllocationString(copy_thunk.dst_buffer_shape().slice()),
            ")")},
       {"{{INPUT_PTR}}",
        absl::StrCat(
            "reinterpret_cast<char*>(",
            GetBufferAllocationString(copy_thunk.src_buffer_shape().slice()),
            ")")},
       {"{{SIZE}}",
        absl::StrCat(copy_thunk.src_buffer_shape().slice().size())}});
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetSortThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_sort_thunk()) {
    return xla::Internal(
        "Sort thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::SortThunkProto& sort_thunk = thunk.sort_thunk();

  std::vector<std::string> buffers_to_sort;
  buffers_to_sort.reserve(sort_thunk.inputs_shapes_size());

  std::vector<int32_t> values_primitive_type_size_in_bytes;
  values_primitive_type_size_in_bytes.reserve(sort_thunk.inputs_shapes_size());
  for (const auto& buffer_proto : sort_thunk.inputs_shapes()) {
    buffers_to_sort.push_back(
        absl::StrCat("reinterpret_cast<char*>(",
                     GetBufferAllocationString(buffer_proto.slice()), ")"));
    values_primitive_type_size_in_bytes.push_back(
        xla::ShapeUtil::ByteSizeOfPrimitiveType(
            buffer_proto.shape().element_type()));
  }
  absl::string_view sort_thunk_invocation_format = R"(
     // Sort Thunk
     {
       std::vector<char*> values = {
         {{BUFFERS_TO_SORT}}
       };
       std::vector<int32_t> values_primitive_type_size_in_bytes = {
         {{VALUES_PRIMITIVE_TYPE_SIZE_IN_BYTES}}
       };

       __xla_cpu_runtime_KeyValueSort(
         {{HIGHER_DIMENSIONS}}, {{SORT_DIMENSION_ELEMENTS}}, {{LOWER_DIMENSIONS}},
         values.data(),
         int32_t(values.size()),
         values_primitive_type_size_in_bytes.data(),
         /*is_stable=*/{{IS_STABLE}},
         reinterpret_cast<char*>(run_options),
         /*prof_counters=*/nullptr,
         reinterpret_cast<void(*)(char*, char*, char**, char**, int64_t*)>({{SORT_FUNCTION_NAME}}));
     })";

  TF_ASSIGN_OR_RETURN(
      auto keys_shape,
      xla::Shape::FromProto(sort_thunk.inputs_shapes(0).shape()));

  // Normalize the shape and the dimension to sort.
  xla::Shape normalized_keys_shape =
      xla::ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
          keys_shape);
  auto logical_to_physical =
      xla::LayoutUtil::MakeLogicalToPhysical(keys_shape.layout());
  TF_RET_CHECK(sort_thunk.dimension() < logical_to_physical.size());
  int64_t physical_dimension_to_sort =
      logical_to_physical[sort_thunk.dimension()];

  int64_t sort_dimension_elements =
      normalized_keys_shape.dimensions(physical_dimension_to_sort);
  int64_t higher_dimensions = 1;
  for (int64_t i = 0; i < physical_dimension_to_sort; ++i) {
    higher_dimensions *= normalized_keys_shape.dimensions(i);
  }
  int64_t lower_dimensions = 1;
  for (int64_t i = normalized_keys_shape.dimensions().size() - 1;
       i > physical_dimension_to_sort; --i) {
    lower_dimensions *= normalized_keys_shape.dimensions(i);
  }
  return absl::StrReplaceAll(
      sort_thunk_invocation_format,
      {
          {"{{HIGHER_DIMENSIONS}}", absl::StrCat(higher_dimensions)},
          {"{{SORT_DIMENSION_ELEMENTS}}",
           absl::StrCat(sort_dimension_elements)},
          {"{{LOWER_DIMENSIONS}}", absl::StrCat(lower_dimensions)},
          {"{{SORT_FUNCTION_NAME}}", sort_thunk.comparator_name()},
          {"{{BUFFERS_TO_SORT}}", absl::StrJoin(buffers_to_sort, ", ")},
          {"{{VALUES_PRIMITIVE_TYPE_SIZE_IN_BYTES}}",
           absl::StrJoin(values_primitive_type_size_in_bytes, ", ")},
          {"{{IS_STABLE}}", sort_thunk.is_stable() ? "true" : "false"},
      });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetTopKThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_top_k_thunk()) {
    return xla::Internal(
        "TopK thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::TopKThunkProto& topk_thunk_proto = thunk.top_k_thunk();

  absl::string_view topk_thunk_invocation_format = R"(
     // TopK Thunk
     {
    __xla_cpu_runtime_TopKF32({{BATCH_SIZE}}, {{INPUT_SIZE}}, {{K}},
                              reinterpret_cast<const float*>({{VALUES_PTR}}),
                              reinterpret_cast<float*>({{OUTPUT_PTR}}),
                              reinterpret_cast<int32_t*>({{INDICES_PTR}}));
     })";

  return absl::StrReplaceAll(
      topk_thunk_invocation_format,
      {
          {"{{BATCH_SIZE}}", absl::StrCat(topk_thunk_proto.batch_size())},
          {"{{INPUT_SIZE}}", absl::StrCat(topk_thunk_proto.input_size())},
          {"{{K}}", absl::StrCat(topk_thunk_proto.k())},
          {"{{VALUES_PTR}}",
           GetBufferAllocationString(topk_thunk_proto.values_buffer())},
          {"{{OUTPUT_PTR}}",
           GetBufferAllocationString(topk_thunk_proto.output_buffer())},
          {"{{INDICES_PTR}}",
           GetBufferAllocationString(topk_thunk_proto.indices_buffer())},
      });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetKernelThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_kernel_thunk()) {
    return xla::Internal(
        "Kernel thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::KernelThunkProto& kernel_thunk = thunk.kernel_thunk();

  auto get_args_initializer_as_string =
      [](const xla::cpu::KernelThunkProto& kernel_thunk) -> std::string {
    std::vector<std::string> args_initializer;
    for (const auto& buffer_proto : kernel_thunk.arguments_buffers()) {
      args_initializer.push_back(absl::StrCat(
          "XLA_CPU_KernelArg{", GetBufferAllocationString(buffer_proto), ", ",
          buffer_proto.size(), "}"));
    }
    for (const auto& buffer_proto : kernel_thunk.results_buffers()) {
      args_initializer.push_back(absl::StrCat(
          "XLA_CPU_KernelArg{", GetBufferAllocationString(buffer_proto), ", ",
          buffer_proto.size(), "}"));
    }
    return absl::StrCat("{", absl::StrJoin(args_initializer, ", "), "}");
  };

  // Execute in block so we don't have to worry about naming for now
  absl::string_view kernel_invocation_format = R"(
     // Kernel Thunk
     {
       std::array<XLA_CPU_KernelArg, {{NUM_ARGS}}> args = {{ARGS_INITIALIZER}};
       XLA_CPU_NumWorkGroups kernel_thread_dims = {
           {{THREAD_DIM_X}},
           {{THREAD_DIM_Y}},
           {{THREAD_DIM_Z}},
       };

       for (uint64_t z = 0; z < {{THREAD_DIM_Z}}; ++z) {
         for (uint64_t y = 0; y < {{THREAD_DIM_Y}}; ++y) {
           for (uint64_t x = 0; x < {{THREAD_DIM_X}}; ++x) {
             XLA_CPU_WorkGroupId kernel_thread = {x, y, z};

             XLA_CPU_KernelCallFrame call_frame = {
                 &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

             XLA_CPU_KernelError* error = (*{{KERNEL_NAME}})(&call_frame);

             if (ABSL_PREDICT_FALSE(error != nullptr)) {
               return false;
             }
           }
         }
       }
     })";

  return absl::StrReplaceAll(
      kernel_invocation_format,
      {
          {"{{NUM_ARGS}}",
           absl::StrCat(kernel_thunk.arguments_buffers().size() +
                        kernel_thunk.results_buffers().size())},
          {"{{ARGS_INITIALIZER}}",
           get_args_initializer_as_string(kernel_thunk)},
          {"{{THREAD_DIM_X}}", absl::StrCat(kernel_thunk.num_workgroups().x())},
          {"{{THREAD_DIM_Y}}", absl::StrCat(kernel_thunk.num_workgroups().y())},
          {"{{THREAD_DIM_Z}}", absl::StrCat(kernel_thunk.num_workgroups().z())},
          {"{{KERNEL_NAME}}", kernel_thunk.kernel_name()},
      });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetConditionalThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_conditional_thunk()) {
    return xla::Internal(
        "Conditional thunk was expected when getting thunk run "
        "implementation.");
  }
  const xla::cpu::ConditionalThunkProto& conditional_thunk =
      thunk.conditional_thunk();

  std::vector<std::string> conditional_thunk_branches;
  conditional_thunk_branches.reserve(conditional_thunk.branch_sequences_size());
  for (const auto& branch_sequence : conditional_thunk.branch_sequences()) {
    TF_ASSIGN_OR_RETURN(conditional_thunk_branches.emplace_back(),
                        ThunkSpecificRunImplFromThunkSequence(branch_sequence));
  }

  absl::string_view branch_execution_format = R"(
         case {{CASE_INDEX}}: {
           {{BRANCH_EXECUTION}}
           break;
         }
     )";

  std::vector<std::string> branch_execution_impls;
  branch_execution_impls.reserve(conditional_thunk_branches.size());

  for (size_t i = 0; i < conditional_thunk_branches.size(); ++i) {
    branch_execution_impls.push_back(absl::StrReplaceAll(
        branch_execution_format,
        {
            {"{{CASE_INDEX}}", absl::StrCat(i)},
            {"{{BRANCH_EXECUTION}}", conditional_thunk_branches[i]},
        }));
  }

  absl::string_view conditional_thunk_invocation_format = R"(
     // Conditional Thunk
     {
       size_t branch_index = {{BRANCH_INDEX}};
       CHECK(branch_index < {{NUM_BRANCHES}}) << "branch_index is out of bounds";
       switch (branch_index) {
         {{BRANCH_EXECUTIONS}}
       }
     })";

  auto get_branch_index =
      [](const xla::buffer_assignment::BufferAllocationSliceProto&
             branch_index_buffer) -> absl::StatusOr<std::string> {
    if (branch_index_buffer.size() == sizeof(bool)) {
      return absl::StrCat("*reinterpret_cast<bool*>(",
                          GetBufferAllocationString(branch_index_buffer),
                          ") ? 0 : 1");
    }
    if (branch_index_buffer.size() == sizeof(int32_t)) {
      return absl::StrCat("*reinterpret_cast<int32_t*>(",
                          GetBufferAllocationString(branch_index_buffer), ")");
    }

    return xla::Internal("Unsupported branch index buffer size %d",
                         branch_index_buffer.size());
  };

  TF_ASSIGN_OR_RETURN(
      std::string branch_index,
      get_branch_index(conditional_thunk.branch_index_buffer()));

  return absl::StrReplaceAll(
      conditional_thunk_invocation_format,
      {
          {"{{BRANCH_INDEX}}", branch_index},
          {"{{NUM_BRANCHES}}", absl::StrCat(branch_execution_impls.size())},
          {"{{BRANCH_EXECUTIONS}}",
           absl::StrJoin(branch_execution_impls, "\n")},
      });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetForLoopThunkRunImpl(
    const xla::cpu::WhileThunkProto& while_thunk) {
  if (!while_thunk.has_trip_count()) {
    return xla::Internal("While thunk is missing trip count.");
  }
  int64_t trip_count = while_thunk.trip_count().value();

  absl::string_view for_loop_thunk_invocation_format = R"(
     // For Loop Thunk
     {
       for (int64_t loop_counter = 0; loop_counter < {{TRIP_COUNT}}; ++loop_counter) {
         {{BODY_EXECUTION}};
       }
     }
     )";

  TF_ASSIGN_OR_RETURN(
      std::string body_execution,
      ThunkSpecificRunImplFromThunkSequence(while_thunk.body_sequence()));

  return absl::StrReplaceAll(for_loop_thunk_invocation_format,
                             {
                                 {"{{TRIP_COUNT}}", absl::StrCat(trip_count)},
                                 {"{{BODY_EXECUTION}}", body_execution},
                             });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetWhileThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_while_thunk()) {
    return xla::Internal(
        "While thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::WhileThunkProto& while_thunk = thunk.while_thunk();

  if (!while_thunk.has_trip_count()) {
    return xla::Internal("Only while thunks with a trip count are supported.");
  }

  return GetForLoopThunkRunImpl(while_thunk);
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::CppDataTypeFromXlaType(
    xla::PrimitiveType xla_type) {
  switch (xla_type) {
    case xla::F16:
      return "Eigen::half";
    case xla::F32:
      return "float";
    case xla::F64:
      return "double";
    case xla::C64:
      return "std::complex<float>";
    case xla::C128:
      return "std::complex<double>";
    case xla::S32:
      return "int32_t";
    default:
      return xla::Internal("Unsupported xla type: %d", xla_type);
  }
}

}  // namespace tfcompile
}  // namespace tensorflow
