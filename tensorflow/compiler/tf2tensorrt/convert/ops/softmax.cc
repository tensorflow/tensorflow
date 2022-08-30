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

class ConvertSoftmax : public OpConverterBase<ConvertSoftmax> {
 public:
  explicit ConvertSoftmax(const OpConverterParams *params)
      : OpConverterBase<ConvertSoftmax>(params) {}

  static constexpr std::array<DataType, 3> AllowedDataTypes() {
    return {DataType::DT_FLOAT, DataType::DT_HALF};
  }

  static constexpr std::array<InputArgSpec, 1> InputSpec() {
    return std::array<InputArgSpec, 1>{
        InputArgSpec::Create("logits", TrtInputArg::kTensor)};
  }

  Status Validate() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;

    ITensorProxyPtr logits_tensor = inputs.at(0).tensor();
    const int num_trt_dims = logits_tensor->getDimensions().nbDims;
    if (!num_trt_dims && params.use_implicit_batch) {
      return errors::InvalidArgument(
          "TensorRT Softmax cannot apply on the batch dimension");
    }
    return Status::OK();
  }

  Status Convert() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    const auto &node_def = params.node_def;

    ITensorProxyPtr logits_tensor = inputs.at(0).tensor();
    const int num_trt_dims = logits_tensor->getDimensions().nbDims;

    // Perform Softmax operation:
    nvinfer1::ISoftMaxLayer *layer =
        params.converter->network()->addSoftMax(*logits_tensor->trt_tensor());
    TFTRT_RETURN_ERROR_IF_NULLPTR(layer, node_def.name());
    params.converter->SetLayerName(layer, node_def);
    // Tensorflow SoftMax applies softmax operation over the last dimension.
    layer->setAxes(1 << (num_trt_dims - 1));

    ITensorProxyPtr output_tensor = layer->getOutput(0);
    params.outputs->push_back(TRT_TensorOrWeights(output_tensor));
    return Status::OK();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertSoftmax>(),
                                  "Softmax");

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
