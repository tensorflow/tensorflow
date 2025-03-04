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

class ConvertTile : public OpConverterBase<ConvertTile> {
 public:
  explicit ConvertTile(const OpConverterParams *params)
      : OpConverterBase<ConvertTile>(
            params,
            {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32}) {}

  static constexpr std::array<InputArgSpec, 2> InputSpec() {
    return std::array<InputArgSpec, 2>{
        InputArgSpec::Create("input_tensor", TrtInputArg::kBoth),
        InputArgSpec::Create("weight", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;

    const auto &repl = inputs.at(1);
    if (params.use_implicit_batch && repl.is_tensor()) {
      return errors::InvalidArgument(
          "Conversion for Tile is not implemented for multipliers "
          "passed as a tensor in implicit batch mode.");
    }

    nvinfer1::DataType dtype;
    const int *multiplies;
    if (repl.is_weights()) {
      TFTRT_CHECK_SHAPE_TENSOR(repl.weights().GetTensor());
      dtype = repl.weights().TrtDType();
      multiplies = repl.weights().GetPointer<int>();
    } else {
      dtype = repl.tensor()->getType();
      multiplies = nullptr;
    }

    const auto &node = params.node_def;
    TF_RETURN_IF_ERROR(check_type(dtype, nvinfer1::DataType::kINT32, node, 1));

    const auto dims = inputs.at(0).GetTrtDims();
    const auto nb_dims =
        dims.nbDims +
        (params.use_implicit_batch && inputs.at(0).is_tensor() ? 1 : 0);
    if (multiplies) {
      const int mult_numb = repl.weights().count();
      if (mult_numb != nb_dims) {
        return errors::InvalidArgument(
            "The length of the replication vector (", mult_numb,
            ") of the Tile operation in '", node.name(),
            "' is expected to be equal to the rank of the input vector (",
            nb_dims, ").");
      }

      if (std::any_of(multiplies, multiplies + nb_dims,
                      [](int i) { return i <= 0; })) {
        const auto &mul = absl::StrJoin(multiplies, multiplies + nb_dims, ", ");
        return errors::InvalidArgument(
            "All replications of the Tile operation in '", node.name(),
            "' should be positive, got (", mul, ").");
      }

      if (params.use_implicit_batch && multiplies[0] > 1) {
        return errors::Unimplemented(
            "The Tile operation along the batch dimension in '", node.name(),
            "' is not implemented.");
      }
    } else {
      const auto &repl_dims = repl.GetTrtDims();
      if (repl_dims.nbDims != 1) {
        return errors::InvalidArgument(
            "When replications are defined as a tensor, that tensor must be "
            "1-dimensional. Got ",
            repl_dims.nbDims, "-dimensional tensor.");
      }

      // Check the number of elements in multiplyer for tensors with non-dynamic
      // shape
      if (repl_dims.d[0] >= 0 && repl_dims.d[0] != nb_dims) {
        return errors::InvalidArgument(
            "When replications are defined as a tensor, "
            "the number of its elements (",
            repl_dims.d[0], ") must be equal to the rank of the input tensor (",
            nb_dims, ").");
      }
    }

    return OkStatus();
  }

  Status Convert() {
    const auto &params = *this->params_;
    const auto &inputs = params.inputs;
    auto *converter = params.converter;
    auto *network = converter->network();
    const auto &tensor = inputs.at(0);
    const auto &replics = inputs.at(1);
    const auto dims = tensor.GetTrtDims();
    const auto nb_dims = dims.nbDims;

    nvinfer1::Dims output_size{nb_dims, {1}};
    bool dynamic_flag = replics.is_tensor() || !HasStaticShape(dims);

    if (!dynamic_flag) {
      // If input0 is a tensor, and we're in implicit batch mode, then we need
      // dim_offset.
      const auto dim_offset =
          params.use_implicit_batch && tensor.is_tensor() ? 1 : 0;
      const auto *input_size = dims.d;
      const int *pReplics = replics.weights().GetPointer<int>() + dim_offset;
      for (int i = 0; i < nb_dims; i++)
        output_size.d[i] = pReplics[i] * input_size[i];
    }

    StatusOr<TRTNetworkBuilder> builder;
    if (tensor.is_weights() || (dynamic_flag && replics.is_weights())) {
      builder =
          TRTNetworkBuilder::Create(converter->network(), params.weight_store);
      TRT_ENSURE_OK(builder);
    }

    ITensorProxyPtr input_tensor;
    if (tensor.is_weights()) {
      StatusOr<nvinfer1::IConstantLayer *> weights_const =
          builder->WeightsToConstant(tensor.weights().GetTrtWeights(), dims);
      TRT_ENSURE_PTR_OK(weights_const);
      input_tensor = (*weights_const)->getOutput(0);
    } else {
      input_tensor = tensor.tensor();
    }

    auto &input_trt_tensor = *input_tensor->trt_tensor();
    nvinfer1::ITensor *target_shape = nullptr;
    if (dynamic_flag) {
      nvinfer1::ITensor *mult;
      if (replics.is_weights()) {
        StatusOr<nvinfer1::IConstantLayer *> weights_const =
            builder->WeightsToConstant(replics.weights().GetTrtWeights(),
                                       replics.GetTrtDims());
        TRT_ENSURE_PTR_OK(weights_const);
        mult = (*weights_const)->getOutput(0);
      } else {
        const ITensorProxyPtr multiplies = replics.tensor()->trt_tensor();
        mult = multiplies->trt_tensor();
      }

      nvinfer1::ITensor *shape =
          network->addShape(input_trt_tensor)->getOutput(0);
#if IS_TRT_VERSION_GE(10, 0, 0, 0)
      // TODO(benbarsdell): Casting to int32 makes this match the pre-TRT10
      // behavior, but it would be better to instead cast all the other int32
      // tensors to int64.
      shape =
          network->addCast(*shape, nvinfer1::DataType::kINT32)->getOutput(0);
#endif
      target_shape = network
                         ->addElementWise(*shape, *mult,
                                          nvinfer1::ElementWiseOperation::kPROD)
                         ->getOutput(0);
    }

    nvinfer1::Dims start{nb_dims, {}};
    DimsAdapter stride(std::vector<int>(nb_dims, 1));
    auto layer = network->addSlice(input_trt_tensor, start, output_size,
                                   stride.AsTrtDims());
#if !IS_TRT_VERSION_GE(10, 0, 0, 0)
    layer->setMode(nvinfer1::SliceMode::kWRAP);
#else
    layer->setMode(nvinfer1::SampleMode::kWRAP);
#endif
    if (target_shape) layer->setInput(2, *target_shape);

    converter->SetLayerName(layer, params.node_def.name(), "to_tile");
    ITensorProxyPtr output_tensor = layer->getOutput(0);
    if (tensor.is_weights() && params.use_implicit_batch) {
      // Reshape output tensor by removing first dimension.
      DimsAdapter adap(output_tensor->getDimensions());
      TF_RETURN_IF_ERROR(adap.RemoveBatchDimension());

      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, TRT_TensorOrWeights(output_tensor),
          adap.AsTrtDims(), false, &output_tensor, params.node_def));
    }

    AddOutput(TRT_TensorOrWeights(output_tensor));
    return OkStatus();
  }
};

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertTile>(), "Tile");

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
