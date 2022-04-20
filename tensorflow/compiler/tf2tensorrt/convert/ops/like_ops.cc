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

template <int V>
class ConvertLikeOps : public OpConverterBase<ConvertLikeOps<V>> {
 public:
  explicit ConvertLikeOps(OpConverterParams *params)
      : OpConverterBase<ConvertLikeOps<V>>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("input", TrtInputArg::kBoth),
    };
  }
  Status Validate() {
    const auto &params = *this->params_;

    const std::string op_name = V == 0 ? "ZerosLike" : "OnesLike";
    if (params.use_implicit_batch) {
      return errors::Unimplemented("Conversion for " + op_name +
                                   " is not implemented in"
                                   " implicit batch mode");
    }
    const auto &inputs = params.inputs;
    if (inputs.size() != 1) {
      return errors::InvalidArgument(op_name, " expects 1 input, but received ",
                                     inputs.size());
    }
    return Status::OK();
  }

  Status Convert() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    auto *converter = params.converter;
    auto *network = converter->network();

    const TRT_TensorOrWeights &input = inputs.at(0);

    nvinfer1::Dims dims{0};

    nvinfer1::DataType value_type = input.TrtDType();

    std::vector<int> value_input_dims_data = {1};
    DimsAdapter value_input_dims(value_input_dims_data);
    StatusOr<TRT_ShapedWeights> value_weights =
        params.weight_store->GetTempWeights(value_type, value_input_dims);
    TF_RETURN_IF_ERROR(value_weights.status());
    TF_RETURN_IF_ERROR(value_weights->SetValues(V));
    TRT_TensorOrWeights value_input(value_weights.ValueOrDie());

    int is_dims_static = true;
    ITensorProxyPtr dims_input_tensor;
    if (!HasStaticShape(input.GetTrtDims())) {
      is_dims_static = false;
      auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
      StatusOr<nvinfer1::IShapeLayer *> shape_layer =
          builder->Shape(input.tensor()->trt_tensor());
      TF_RETURN_IF_ERROR(shape_layer.status());
      dims_input_tensor = (*shape_layer)->getOutput(0);
    } else {
      dims = input.GetTrtDims();
    }

    TRT_TensorOrWeights dims_input(dims_input_tensor);
    auto builder = TRTNetworkBuilder::Create(network, params.weight_store);
    StatusOr<nvinfer1::ILayer *> layer =
        builder->AddFill(value_input, dims_input, true, is_dims_static,
                         input.GetTrtDims().nbDims, dims);
    ITensorProxyPtr output_tensor = (*layer)->getOutput(0);
    this->AddOutput(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<0>>(),
                                  "zeros_like");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<1>>(),
                                  "ones_like");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<0>>(),
                                  "ZerosLike");
REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertLikeOps<1>>(),
                                  "OnesLike");

#endif  // IS_TRT_VERSION_GE(8, 2, 0, 0)

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
