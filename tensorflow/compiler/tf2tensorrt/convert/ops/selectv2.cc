/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

#if IS_TRT_VERSION_GE(8, 2, 0, 0)
/*  The ConvertSelectV2 is working only for cond_input passed as a boolean
 *  tensor, which could be created only for TRT >= 8.2
 */
class ConvertSelectV2 : public OpConverterBase<ConvertSelectV2> {
 public:
  explicit ConvertSelectV2(const OpConverterParams* params)
      : OpConverterBase<ConvertSelectV2>(
            params,
            {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}) {}

  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return std::array<InputArgSpec, 3>{
        InputArgSpec::Create("cond", TrtInputArg::kBoth),
        InputArgSpec::Create("then", TrtInputArg::kBoth),
        InputArgSpec::Create("else", TrtInputArg::kBoth)};
  }

  Status Validate() {
    TF_RETURN_IF_ERROR(NotSupportedInImplicitBatch());

    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    const auto& cond_input = inputs.at(0);
    const auto& node = params.node_def;
    TF_RETURN_IF_ERROR(
        check_type(cond_input.TrtDType(), nvinfer1::DataType::kBOOL, node));

    if (cond_input.is_weights()) {
      return errors::InvalidArgument(bool_weight_error_msg(node));
    }

    const auto type_then = inputs[1].TrtDType();
    const auto type_else = inputs[2].TrtDType();
    if (type_then != type_else && (type_then == nvinfer1::DataType::kINT32 ||
                                   type_else == nvinfer1::DataType::kINT32)) {
      // Both or none of (type_then, type_else) should be equal to kINT32.
      return errors::InvalidArgument(
          then_else_dtypes_error_msg(type_then, type_else, node));
    }

    nvinfer1::Dims broadcasted_dims[3];
    for (int i = 1; i < 3; i++) {
      TF_RETURN_IF_ERROR(GetTrtBroadcastShape(cond_input, inputs.at(i), true,
                                              false, broadcasted_dims,
                                              broadcasted_dims + i));
    }

    for (int i = 0; i < tensor_.size(); i++) {
      // This will also convert constants to tensors.
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, inputs.at(i), broadcasted_dims[i],
          params.validation_only, &tensor_[i], node, i));
    }

    return Status::OK();
  }

  Status Convert() {
    const auto& params = *this->params_;
    auto* converter = params.converter;

    nvinfer1::ISelectLayer* select_layer = converter->network()->addSelect(
        *tensor_[0]->trt_tensor(),  // cond_tensor
        *tensor_[1]->trt_tensor(),  // then_tensor
        *tensor_[2]->trt_tensor()   // else_tensor
    );

    converter->SetLayerName(select_layer, params.node_def.name(), "selectv2");
    AddOutput(TRT_TensorOrWeights(select_layer->getOutput(0)));
    return Status::OK();
  }

 private:
  std::array<ITensorProxyPtr, 3> tensor_{nullptr, nullptr, nullptr};
};

std::string bool_weight_error_msg(const NodeDef& node_def) {
  return "The boolean parameter '" + node_def.input(0) + "' of the " +
         node_def.op() + " operation in " + node_def.name() +
         " cannot be passed as a weight in TRT version 8.4.";
}

std::string then_else_dtypes_error_msg(nvinfer1::DataType type_then,
                                       nvinfer1::DataType type_else,
                                       const NodeDef& node) {
  return "DataTypes (" + DebugString(type_then) + ", " +
         DebugString(type_else) + ") of parameters (" + node.input(1) + ", " +
         node.input(2) + ") of " + node.op() + " operation in " + node.name() +
         " are incompatible.";
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertSelectV2>(),
                                  "SelectV2");
#endif

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
