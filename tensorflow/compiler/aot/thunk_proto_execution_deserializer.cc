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
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/backends/cpu/runtime/convolution_lib.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace tensorflow {
namespace tfcompile {

namespace {

std::string GetBufferAllocationString(
    const xla::buffer_assignment::BufferAllocationSliceProto& slice) {
  return absl::StrCat("reinterpret_cast<std::byte*>(buffer_table()[",
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
      case xla::cpu::ThunkProto::kCopyThunk: {
        TF_ASSIGN_OR_RETURN(thunk_run_impls.emplace_back(),
                            GetCopyThunkRunImpl(thunk));
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

  absl::string_view dot_thunk_invocation_format = R"(
     // Dot Thunk
     {
         if (run_options()->intra_op_thread_pool() != nullptr) {
           {{MATMUL_FUNCTION}}(
            run_options(), {{OUTPUT_PTR}}, {{LHS_PTR}}, {{RHS_PTR}},
            {{M}}, {{N}}, {{K}}, {{TRANSPOSE_LHS}}, {{TRANSPOSE_RHS}});
         } else {
           {{SINGLE_THREADED_MATMUL_FUNCTION}}(
            nullptr, {{OUTPUT_PTR}}, {{LHS_PTR}}, {{RHS_PTR}},
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

  auto lhs_shape = xla::Shape(dot_thunk.lhs_buffer_shape().shape());
  auto rhs_shape = xla::Shape(dot_thunk.rhs_buffer_shape().shape());
  auto out_shape = xla::Shape(dot_thunk.out_buffer_shape().shape());

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

  return absl::StrReplaceAll(
      dot_thunk_invocation_format,
      {{"{{MATMUL_FUNCTION}}", matmul_function},
       {"{{SINGLE_THREADED_MATMUL_FUNCTION}}", single_threaded_matmul_function},
       {"{{OUTPUT_PTR}}", output_ptr},
       {"{{LHS_PTR}}", lhs_ptr},
       {"{{RHS_PTR}}", rhs_ptr},
       {"{{M}}", absl::StrCat(m)},
       {"{{N}}", absl::StrCat(n)},
       {"{{K}}", absl::StrCat(k)},
       {"{{TRANSPOSE_LHS}}", transpose_lhs ? "true" : "false"},
       {"{{TRANSPOSE_RHS}}", transpose_rhs ? "true" : "false"}});
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

  absl::string_view convolution_thunk_invocation_format = R"(
     // Convolution Thunk
     {
         if (run_options()->intra_op_thread_pool() != nullptr) {
           {{CONVOLUTION_FUNCTION}}(
             run_options(),
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
     })";

  return absl::StrReplaceAll(
      convolution_thunk_invocation_format,
      {{"{{CONVOLUTION_FUNCTION}}", convolution_function},
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
        absl::StrCat(canonical_dims.feature_group_count)}});
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

  // NOTE(basioli): Slices are not needed here, we only use this class to
  // invoke GetConvolutionCanonicalDims.
  xla::cpu::ConvolutionSlices slices{
      /*input_buffer =*/{},
      /*input_shape =*/
      xla::Shape(convolution_thunk.input_buffer_shape().shape()),
      /*kernel_buffer =*/{},
      /*kernel_shape =*/
      xla::Shape(convolution_thunk.kernel_buffer_shape().shape()),
      /*output_buffer =*/{},
      /*output_shape =*/
      xla::Shape(convolution_thunk.output_buffer_shape().shape()),
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
         rng_states_[{{RNG_STATE_INDEX}}].GetAndUpdateState({{RNG_STATE_PTR}});
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
       XLA_CPU_KernelThreadDim kernel_thread_dims = {
           {{THREAD_DIM_X}},
           {{THREAD_DIM_Y}},
           {{THREAD_DIM_Z}},
       };

       for (uint64_t z = 0; z < {{THREAD_DIM_Z}}; ++z) {
         for (uint64_t y = 0; y < {{THREAD_DIM_Y}}; ++y) {
           for (uint64_t x = 0; x < {{THREAD_DIM_X}}; ++x) {
             XLA_CPU_KernelThread kernel_thread = {x, y, z};

             XLA_CPU_KernelCallFrame call_frame = {
                 &kernel_thread_dims, &kernel_thread, args.size(), args.data()};

             XLA_CPU_KernelError* error = (*{{KERNEL_NAME}})(&call_frame);

             if (ABSL_PREDICT_FALSE(error != nullptr)) {
               return false;
             }
           }
         }
       }
     }
     )";

  return absl::StrReplaceAll(
      kernel_invocation_format,
      {
          {"{{NUM_ARGS}}",
           absl::StrCat(kernel_thunk.arguments_buffers().size() +
                        kernel_thunk.results_buffers().size())},
          {"{{ARGS_INITIALIZER}}",
           get_args_initializer_as_string(kernel_thunk)},
          {"{{THREAD_DIM_X}}", absl::StrCat(kernel_thunk.thread_dim().x())},
          {"{{THREAD_DIM_Y}}", absl::StrCat(kernel_thunk.thread_dim().y())},
          {"{{THREAD_DIM_Z}}", absl::StrCat(kernel_thunk.thread_dim().z())},
          {"{{KERNEL_NAME}}", kernel_thunk.kernel_name()},
      });
}

absl::StatusOr<std::string>
ThunkProtoExecutionDeserializer::GetCopyThunkRunImpl(
    const xla::cpu::ThunkProto& thunk) {
  if (!thunk.has_copy_thunk()) {
    return xla::Internal(
        "Copy thunk was expected when getting thunk run implementation.");
  }
  const xla::cpu::CopyThunkProto& copy_thunk = thunk.copy_thunk();

  if (!xla::ShapeUtil::Equal(
          xla::Shape(copy_thunk.src_buffer_shape().shape()),
          xla::Shape(copy_thunk.dst_buffer_shape().shape()))) {
    return xla::Internal("Source and destination shapes must be equal.");
  }

  absl::string_view copy_invocation_format = R"(
     // Copy Thunk
     {
       std::memcpy({{DST_BUFFER}},
                   {{SRC_BUFFER}},
                   {{SRC_BUFFER_SIZE}});
     }
     )";

  return absl::StrReplaceAll(
      copy_invocation_format,
      {
          {"{{DST_BUFFER}}",
           GetBufferAllocationString(copy_thunk.dst_buffer_shape().slice())},
          {"{{SRC_BUFFER}}",
           GetBufferAllocationString(copy_thunk.src_buffer_shape().slice())},
          {"{{SRC_BUFFER_SIZE}}",
           absl::StrCat(copy_thunk.src_buffer_shape().slice().size())},
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
    default:
      return xla::Internal("Unsupported xla type: %d", xla_type);
  }
}

}  // namespace tfcompile
}  // namespace tensorflow
