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
 *  tensor, which could be created only for TRT >= 8.2.
 */
class ConvertSelectBase : public OpConverterBase<ConvertSelectBase> {
 public:
  explicit ConvertSelectBase(const OpConverterParams* params,
                             const std::string& layer_name)
      : OpConverterBase<ConvertSelectBase>(
            params,
            {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}),
        layer_name_(layer_name) {}

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
    const auto& i_cond = inputs.at(0);
    const auto& node = params.node_def;
    TF_RETURN_IF_ERROR(
        check_type(i_cond.TrtDType(), nvinfer1::DataType::kBOOL, node));

    if (i_cond.is_weights()) {
      // According to
      // https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#constant-layer
      // Boolean weights are not supported in TRT version 8.4.
      return errors::InvalidArgument(bool_weight_error_msg(node));
    }

    const auto& i_then = inputs.at(1);
    const auto& i_else = inputs.at(2);
    const auto type_then = i_then.TrtDType();
    const auto type_else = i_else.TrtDType();
    if (type_then != type_else && (type_then == nvinfer1::DataType::kINT32 ||
                                   type_else == nvinfer1::DataType::kINT32)) {
      // Both or none of (type_then, type_else) should be equal to kINT32.
      return errors::InvalidArgument(
          then_else_dtypes_error_msg(type_then, type_else, node));
    }

    bool cond_is_vector = false;
    const auto& shape_cond = i_cond.GetTrtDims();
    if (layer_name_ == "select") {
      const auto& shape_then = i_then.GetTrtDims();
      const auto& shape_else = i_else.GetTrtDims();
      TF_RETURN_IF_ERROR(compare_shapes(shape_then, shape_else));
      TF_RETURN_IF_ERROR(
          compare_shapes(shape_cond, shape_then, &cond_is_vector));
    }

    nvinfer1::Dims cond_dims(shape_cond);
    if (cond_is_vector) {
      cond_dims.nbDims = i_then.GetTrtDims().nbDims;
      const std::vector<int> ones(cond_dims.d[0], 1);
      std::copy(ones.begin(), ones.end(), cond_dims.d + 1);
    }

    const TRT_TensorOrWeights new_cond(nvinfer1::DataType::kBOOL, cond_dims,
                                       i_cond.batch_size());
    nvinfer1::Dims broadcasted_dims[3];
    for (int i = 1; i < 3; i++) {
      TF_RETURN_IF_ERROR(GetTrtBroadcastShape(new_cond, inputs.at(i), true,
                                              false, broadcasted_dims,
                                              broadcasted_dims + i));
    }

    for (int i = 0; i < tensor_.size(); i++) {
      // This will also convert constants to tensors.
      tensor_[i] = std::make_unique<TRT_TensorOrWeights>(inputs.at(i));
      TF_RETURN_IF_ERROR(
          ApplyBroadcast(tensor_[i], broadcasted_dims[i], this->params_, 0));
    }

    return OkStatus();
  }

  Status Convert() {
    const auto& params = *this->params_;
    auto* converter = params.converter;

    nvinfer1::ISelectLayer* select_layer = converter->network()->addSelect(
        *tensor_[0].get()->as_tensor(params_)->trt_tensor(),  // cond_tensor
        *tensor_[1].get()->as_tensor(params_)->trt_tensor(),  // then_tensor
        *tensor_[2].get()->as_tensor(params_)->trt_tensor()   // else_tensor
    );

    converter->SetLayerName(select_layer, params.node_def.name(), layer_name_);
    AddOutput(TRT_TensorOrWeights(select_layer->getOutput(0)));
    return OkStatus();
  }

 private:
  Status compare_shapes(const nvinfer1::Dims& shape1,
                        const nvinfer1::Dims& shape2,
                        bool* cond_is_vector = nullptr) const {
    const bool then_vs_else = cond_is_vector == nullptr;
    bool same_shapes = shape1 == shape2;
    if (!same_shapes && shape1.nbDims == shape2.nbDims) {
      // We can't check size equivalent when dynamic shapes are involved.
      // In this case, the two shapes should be equal at runtime. Therefore,
      // the shapes still should be considered as equal if at least one of
      // them is a tensor with dynamic shape,
      same_shapes = DynamicShapeInput(this->params_->inputs, then_vs_else);
    }
    if (!same_shapes) {
      if (then_vs_else || !(*cond_is_vector = (shape1.nbDims == 1 &&
                                               shape1.d[0] == shape2.d[0]))) {
        const auto err = input_shapes_error_msg(
            shape1, shape2, this->params_->node_def, then_vs_else);
        return errors::InvalidArgument(err);
      }
    }
    return OkStatus();
  }

  bool DynamicShapeInput(const std::vector<TRT_TensorOrWeights>& inputs,
                         bool then_vs_else) const {
    const int idx = then_vs_else ? 1 : 0;
    for (int i = 0; i < 2; ++i) {
      const auto& input = inputs.at(i + idx);
      if (input.is_tensor() && !HasStaticShape(input.GetTrtDims())) {
        return true;
      }
    }
    return false;
  }

  std::array<std::unique_ptr<TRT_TensorOrWeights>, 3> tensor_;
  const std::string layer_name_;
};

class ConvertSelect : public ConvertSelectBase {
 public:
  explicit ConvertSelect(const OpConverterParams* params)
      : ConvertSelectBase(params, "select") {}
};

class ConvertSelectV2 : public ConvertSelectBase {
 public:
  explicit ConvertSelectV2(const OpConverterParams* params)
      : ConvertSelectBase(params, "selectv2") {}
};

std::string op_node_info(const NodeDef& node) {
  return " of the '" + node.op() + "' operation at the node '" + node.name() +
         "' ";
}

std::string bool_weight_error_msg(const NodeDef& node) {
  return "The boolean parameter '" + node.input(0) + "'" + op_node_info(node) +
         "cannot be passed as a weight in TRT version 8.4.";
}

std::string then_else_dtypes_error_msg(nvinfer1::DataType type_then,
                                       nvinfer1::DataType type_else,
                                       const NodeDef& node) {
  return "DataTypes (" + DebugString(type_then) + ", " +
         DebugString(type_else) + ") of parameters (" + node.input(1) + ", " +
         node.input(2) + ")" + op_node_info(node) + "are incompatible.";
}

std::string input_shapes_error_msg(const nvinfer1::Dims& shape1,
                                   const nvinfer1::Dims& shape2,
                                   const NodeDef& node, bool then_vs_else) {
  const std::string& param_names =
      then_vs_else ? "'then' and 'else'" : "'cond' and 'then'";
  std::string error_msg = "The shapes of the " + param_names + " parameters" +
                          op_node_info(node) + "must be the same";
  if (!then_vs_else) {
    error_msg +=
        " OR 'cond' must be a vector with N elements, "
        "where N is a batch size (the first shape dimension for 'then')";
  }
  return error_msg + ", got " + DebugString(shape1) + " vs. " +
         DebugString(shape2) + ".";
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertSelect>(),
                                  "Select");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertSelectV2>(),
                                  "SelectV2");
#endif

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
