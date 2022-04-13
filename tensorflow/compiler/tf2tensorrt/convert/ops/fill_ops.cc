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

class ConvertFill : public OpConverterBase<ConvertFill> {
 public:
  explicit ConvertFill(OpConverterParams* params)
      : OpConverterBase<ConvertFill>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("dims", TrtInputArg::kBoth),
        InputArgSpec::Create("value", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& params = *this->params_;

    if (params.use_implicit_batch) {
      return errors::Unimplemented(
          "Conversion for Fill is not implemented in"
          "implicit batch mode");
    }

    const auto& inputs = params.inputs;
    const auto& node_def = params.node_def;
    const TRT_TensorOrWeights& dims_input = inputs.at(0);

    nvinfer1::DataType dims_type = dims_input.TrtDType();
    if (dims_type != nvinfer1::DataType::kINT32) {
      return errors::InvalidArgument("The dims parameter of ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " is expected to be of type ",
                                     DebugString(nvinfer1::DataType::kINT32),
                                     " type, got ", DebugString(dims_type));
    }

    int nbDims = dims_input.GetTrtDims().nbDims;
    if (nbDims < 0) {
      return errors::InvalidArgument("The shape of parameter ", node_def.op(),
                                     " operation in ", node_def.name(),
                                     " cannot be partial.");
    }
    return Status::OK();
  }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    auto* converter = params.converter;
    auto* network = converter->network();
    const auto& node_def = params.node_def;

    const bool is_dims_static = inputs[0].is_weights();
    const bool is_value_static = inputs[1].is_weights();

    const TRT_TensorOrWeights& dims_input = inputs.at(0);
    const TRT_TensorOrWeights& value_input = inputs.at(1);

    int nbDims = dims_input.GetTrtDims().d[0];

    nvinfer1::Dims trt_dims{0};
    if (is_dims_static) {
      const auto dims_weights = dims_input.weights();
      DimsAdapter dims_adapter(dims_weights.GetSpan<int32>());
      dims_adapter.TrtDims(&trt_dims);
    }

    // TensorRT IFillLayer requires a rank 0 scalar.
    ITensorProxyPtr scalar_tensor;
    nvinfer1::Dims scalar_dims;
    scalar_dims.nbDims = 0;
    nvinfer1::DataType value_type = value_input.TrtDType();
    if (is_value_static) {
      scalar_tensor =
          converter->CreateConstantLayer(value_input.weights(), scalar_dims);
    } else {
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          converter, value_input, scalar_dims, params.validation_only,
          &scalar_tensor, node_def));
    }

    auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
    nvinfer1::Dims beta_shape{1, {nbDims}};
    StatusOr<nvinfer1::IConstantLayer*> const_layer =
        builder->Constant(0, beta_shape, value_type);
    TF_RETURN_IF_ERROR(const_layer.status());
    ITensorProxyPtr empty_beta_tensor = (*const_layer)->getOutput(0);

    nvinfer1::IFillLayer* layer =
        network->addFill(trt_dims, nvinfer1::FillOperation::kLINSPACE);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    if (!is_dims_static) {
      layer->setInput(0, *dims_input.tensor()->trt_tensor());
    }
    layer->setInput(1, *scalar_tensor->trt_tensor());
    layer->setInput(2, *empty_beta_tensor->trt_tensor());
    converter->SetLayerName(layer, node_def, "fill");
    ITensorProxyPtr output_tensor = layer->getOutput(0);
    AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertFill>(), "Fill");

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
