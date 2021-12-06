/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "tensorflow/compiler/tf2tensorrt/common/utils.h"
#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

int get_spatial_dim_count(string format) {
  // Spatial dimensions are the dimensions besides NC, and here we assume NC
  // always appear in the format string.
  return format.size() - 2;
}

class ConvertDataFormatVecPermute
    : public OpConverterBase<ConvertDataFormatVecPermute> {
 public:
  ConvertDataFormatVecPermute(OpConverterParams* params)
      : OpConverterBase<ConvertDataFormatVecPermute>(params) {}

  struct DataFormatVecPermuteAttributes {
    string dst_format;
    string src_format;
    int x_dim_count;
  };

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return {InputArgSpec::Create("x", TrtInputArg::kBoth)};
  }

  static constexpr std::array<DataType, 1> AllowedDataTypes() {
    return {DataType::DT_INT32};
  }

  Status Validate() {
    const auto& inputs = params_->inputs;
    const auto& node_def = params_->node_def;

    if (params_->use_implicit_batch) {
      return errors::Unimplemented("Implicit batch mode not supported, at ",
                                   node_def.name());
    }

    x_input_ = inputs.at(0);

    // Check input rank.
    const auto x_dims = x_input_.GetTrtDims();
    int input_rank = x_dims.nbDims;
    if (input_rank != 1 && input_rank != 2) {
      return errors::InvalidArgument(
          "Input must be a vector or matrix, but got rank ", input_rank,
          ", at ", node_def.name());
    }

    // Verify and consume node attributes.
    StatusOr<string> dst_format = GetAttrValue<string>("dst_format");
    StatusOr<string> src_format = GetAttrValue<string>("src_format");
    TRT_ENSURE_OK(dst_format);
    TRT_ENSURE_OK(src_format);

    // Check input dims.
    const int full_dim_count = src_format->size();
    const int spatial_dim_count = get_spatial_dim_count(*src_format);
    if (input_rank == 1) {
      if (x_dims.d[0] != spatial_dim_count && x_dims.d[0] != full_dim_count) {
        return errors::InvalidArgument("1D input must be of size ",
                                       spatial_dim_count, " or ",
                                       full_dim_count, ", but got size ",
                                       x_dims.d[0], ", at ", node_def.name());
      }
    } else if (input_rank == 2) {
      if (x_dims.d[0] != spatial_dim_count && x_dims.d[0] != full_dim_count) {
        return errors::InvalidArgument(
            "First dimension of 2D input must be of size ", spatial_dim_count,
            " or ", full_dim_count, ", but got shape (", x_dims.d[0], ", ",
            x_dims.d[1], "), at ", node_def.name());
      }
      if (x_dims.d[1] != 2) {
        return errors::InvalidArgument(
            "Second dimension of 2D input must be of size 2, but got shape (",
            x_dims.d[0], ", ", x_dims.d[1], "), at ", node_def.name());
      }
    }

    // Set custom attributes.
    attrs_.x_dim_count = x_dims.d[0];
    attrs_.dst_format = *dst_format;
    attrs_.src_format = *src_format;

    return Status::OK();
  }

  Status Convert() {
    const auto& node_def = params_->node_def;

    // Copy format strings in case they need to be modified.
    string dst_format = attrs_.dst_format;
    string src_format = attrs_.src_format;
    const int& spatial_dim_count = get_spatial_dim_count(src_format);

    // If the input is a vector of size spatial_dim_count, treat the elements
    // as spatial dimensions.
    if (attrs_.x_dim_count == spatial_dim_count) {
      auto keep_only_spatial_dimensions =
          [spatial_dim_count](string* format_str) -> void {
        auto new_end = std::remove_if(format_str->begin(), format_str->end(),
                                      [spatial_dim_count](const char dim) {
                                        return dim == 'N' || dim == 'C';
                                      });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format);
      keep_only_spatial_dimensions(&dst_format);
    }

    // Create indices for the gather layer and make weights out of them.
    std::vector<int32> dst_indices(attrs_.x_dim_count);
    for (int i = 0; i < attrs_.x_dim_count; ++i) {
      for (int j = 0; j < attrs_.x_dim_count; ++j) {
        if (src_format[i] == dst_format[j]) {
          dst_indices[j] = i;
          break;
        }
      }
    }
    nvinfer1::Dims indices_dims = {1, {attrs_.x_dim_count}};
    auto indices_weights = params_->weight_store->GetTempWeights(
        nvinfer1::DataType::kINT32, indices_dims);
    int32* indices_ptr = indices_weights.GetPointer<int32>();
    std::copy(dst_indices.data(), dst_indices.data() + attrs_.x_dim_count,
              indices_ptr);

    ITensorProxyPtr x_tensor =
        x_input_.is_weights() ? params_->converter->CreateConstantLayer(
                                    x_input_.weights(), x_input_.GetTrtDims())
                              : x_input_.tensor();
    ITensorProxyPtr indices_tensor =
        params_->converter->CreateConstantLayer(indices_weights, indices_dims);

    // Gather layer with 1D indices on axis 0, conserves shape.
    nvinfer1::IGatherLayer* layer = params_->converter->network()->addGather(
        *x_tensor->trt_tensor(), *indices_tensor->trt_tensor(), 0);
    TRT_ENSURE(layer);
    params_->converter->SetLayerName(layer, node_def);

    ITensorProxyPtr output_tensor = layer->getOutput(0);

    params_->outputs->push_back(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }

 private:
  TRT_TensorOrWeights x_input_;
  DataFormatVecPermuteAttributes attrs_{};
};
REGISTER_DEFAULT_TRT_OP_CONVERTER(
    MakeConverterFunction<ConvertDataFormatVecPermute>(),
    {"DataFormatVecPermute"});

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
