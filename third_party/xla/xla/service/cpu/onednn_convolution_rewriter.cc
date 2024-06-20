/* Copyright 2024 The OpenXLA Authors.

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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include "xla/service/cpu/onednn_convolution_rewriter.h"

#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_memory_util.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/status_macros.h"

namespace xla {
namespace cpu {

namespace {
namespace m = match;
}  // namespace

bool OneDnnConvolutionRewriter::ShouldRewrite(const HloInstruction* conv) {
  if (conv->HasControlDependencies()) return false;
  if (!IsSupportedType(conv->shape().element_type())) return false;
  if (conv->batch_group_count() != 1) return false;

  if (conv->operand(1)->opcode() == HloOpcode::kReverse) return false;

  const Shape& inp_shape = conv->operand(0)->shape();
  const Shape& ker_shape = conv->operand(1)->shape();
  const Shape& out_shape = conv->shape();
  if (ShapeUtil::IsZeroElementArray(inp_shape) ||
      ShapeUtil::IsZeroElementArray(ker_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    return false;
  }

  auto dims = conv->window().dimensions().size();
  if (dims >= 4 || dims <= 0) return false;

  if (inp_shape.rank() != ker_shape.rank() ||
      inp_shape.rank() != out_shape.rank()) {
    return false;
  }

  return true;
}

class OneDnnConvolutionRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleConvolution(HloInstruction* conv) override {
    auto pattern = match::Op(&conv).WithOpcode(HloOpcode::kConvolution);
    if (!Match(conv, pattern)) return absl::OkStatus();
    if (!OneDnnConvolutionRewriter::ShouldRewrite(conv)) {
      return absl::OkStatus();
    }

    const Shape& conv_shape = conv->shape();
    auto dims = conv->window().dimensions().size();
    const ConvolutionDimensionNumbers& conv_ddata =
        conv->convolution_dimension_numbers();

    BackendConfig backend_config;
    OneDnnConvolutionConfig* conv_config =
        backend_config.mutable_onednn_conv_config();

    conv_config->set_dims(conv_shape.rank());
    conv_config->set_feature_groups(conv->feature_group_count());
    conv_config->mutable_input()->mutable_data()->set_batch_dim(
        conv_ddata.input_batch_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_input_feature_dim(
        conv_ddata.kernel_input_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_batch_dim(
        conv_ddata.output_batch_dimension());
    conv_config->mutable_input()->mutable_data()->set_feature_dim(
        conv_ddata.input_feature_dimension());
    conv_config->mutable_kernel()->mutable_filter()->set_output_feature_dim(
        conv_ddata.kernel_output_feature_dimension());
    conv_config->mutable_output()->mutable_data()->set_feature_dim(
        conv_ddata.output_feature_dimension());

    const Shape& output_shape = conv->shape();

    for (auto it = conv->window().dimensions().begin();
         it != conv->window().dimensions().end(); it++) {
      if ((*it).padding_low() < 0 || (*it).padding_high() < 0 ||
          (*it).stride() < 0) {
        return absl::OkStatus();
      }
      conv_config->mutable_window()->add_pad_left((*it).padding_low() + 1);
      conv_config->mutable_window()->add_pad_right((*it).padding_high() + 1);
      conv_config->mutable_window()->add_strides((*it).stride() + 1);
      conv_config->mutable_window()->add_window_dilations(
          (*it).window_dilation() + 1);
      if ((*it).base_dilation() != 1 || (*it).window_reversal()) {
        return absl::OkStatus();
      }
    }

    for (int i = 0; i < dims; i++) {
      conv_config->mutable_input()->mutable_data()->add_spatial_dims(
          conv_ddata.input_spatial_dimensions()[i] + 1);
      conv_config->mutable_kernel()->mutable_filter()->add_spatial_dims(
          conv_ddata.kernel_spatial_dimensions()[i] + 1);
      conv_config->mutable_output()->mutable_data()->add_spatial_dims(
          conv_ddata.output_spatial_dimensions()[i] + 1);
    }

    HloInstruction* custom_call =
        conv->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {conv->mutable_operand(0), conv->mutable_operand(1)},
            "__onednn$convolution"));

    TF_RETURN_IF_ERROR(custom_call->set_backend_config(backend_config));
    TF_RETURN_IF_ERROR(ReplaceInstruction(conv, custom_call));
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> OneDnnConvolutionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  OneDnnConvolutionRewriterVisitor visitor;
  return visitor.RunOnModule(module, execution_threads);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
