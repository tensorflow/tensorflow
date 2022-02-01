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

class ConvertUnary : public OpConverterBase<ConvertUnary> {
 public:
  explicit ConvertUnary(OpConverterParams* params)
      : OpConverterBase<ConvertUnary>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("my_unary", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& op = params_->node_def.op();
    if (UnaryOperationMap()->find(op) == UnaryOperationMap()->end()) {
      return errors::Unimplemented("Unary op: ", op, " not supported");
    }

    return CheckInputsWeights(*params_, {{"x", false}});
  }

  Status Convert() {
    const auto &params = *this->params_;
    const auto& node_def = params.node_def;
    auto *converter = params.converter;

    const auto op_pair = UnaryOperationMap()->find(node_def.op());
    ITensorProxyPtr tensor = params.inputs.at(0).tensor();
    nvinfer1::IUnaryLayer* layer = converter->network()->addUnary(
        *tensor->trt_tensor(), op_pair->second);
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    converter->SetLayerName(layer, node_def);

    params.outputs->push_back(TRT_TensorOrWeights(layer->getOutput(0)));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertUnary>(),
                                  GetOperationNames(*UnaryOperationMap()));

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
