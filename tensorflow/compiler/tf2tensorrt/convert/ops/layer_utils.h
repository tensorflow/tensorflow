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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include <type_traits>

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/utils.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/statusor.h"
#include "third_party/tensorrt/NvInfer.h"
#include "third_party/tensorrt/NvInferRuntimeCommon.h"

namespace tensorflow {
namespace tensorrt {

namespace convert {

class NetworkFactoryContext {
 public:
  NetworkFactoryContext(nvinfer1::INetworkDefinition* network,
                        TrtWeightStore* weight_store)
      : network_(network), weight_store_(weight_store) {}

  template <typename T, typename std::enable_if<std::is_base_of<
                            nvinfer1::ILayer, T>::value>::type* = nullptr>
  struct CreationResult {
    T* layer;
    nvinfer1::ITensor* output;
  };

  // Adds an Add operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> Add(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(*lhs, *rhs,
                                          nvinfer1::ElementWiseOperation::kSUM);
    return MakeResult(layer);
  };

  // Adds an Add operation to the network. Note that this unary operations will
  // do an implict float conversion.
  StatusOr<CreationResult<nvinfer1::IUnaryLayer>> Abs(
      nvinfer1::ITensor* input) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(input);
    auto layer = network_->addUnary(*input, nvinfer1::UnaryOperation::kABS);
    return MakeResult(layer);
  }

  // Performs Abs without implict float conversion.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> AbsInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(input);
    auto sign = this->Sign(input);
    TRT_ENSURE_OK(sign);
    return this->Mul(input, sign->output);
  }

  // Returns elementwise sign(x) operation.
  StatusOr<CreationResult<nvinfer1::ISelectLayer>> Sign(
      nvinfer1::ITensor* input) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(input);
    auto zero = this->Constant<int32>(0, input->getDimensions().nbDims);
    TRT_ENSURE_OK(zero);
    auto positive_mask = this->Greater(input, zero->output);
    TRT_ENSURE_OK(positive_mask);
    auto pos_one = this->Constant<int32>(1, input->getDimensions().nbDims);
    TRT_ENSURE_OK(pos_one);
    auto neg_one = this->Constant<int32>(-1, input->getDimensions().nbDims);
    TRT_ENSURE_OK(neg_one);
    return Where(positive_mask->output, pos_one->output, neg_one->output);
  }

  // Adds an Add operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> Sub(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(*lhs, *rhs,
                                          nvinfer1::ElementWiseOperation::kSUB);
    return MakeResult(layer);
  }

  // Adds an Greater operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> Greater(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kGREATER);
    return MakeResult(layer);
  }

  // Adds an Equal operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> Equal(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kEQUAL);
    return MakeResult(layer);
  }

  // Simulates the creation of a tensor of the same shape as "input", but filled
  // with 0's where "input" is non-zero. TensorRT doesn't have dynamic creation
  // functions, so this function multiplies the input by a zero value to side
  // step knowing the shape. For static inputs, this can be optimized using one
  // of the NetworkFactoryContext::Constant function instead.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> ZerosLikeDyamic(
      nvinfer1::ITensor* input) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(input);
    // TODO: add switch statement
    auto zero = this->Constant<int32>(0, input->getDimensions().nbDims);
    TRT_ENSURE_OK(zero);
    return Mul(input, zero->output);
  }

  // Adds an Add operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> FloorDiv(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
    return MakeResult(layer);
  }

  // Abs(Ceiling divide(x/y)) operation for integers
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> AbsCeilDivInt(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto rhs_abs = this->AbsInt(rhs);
    TRT_ENSURE_OK(rhs_abs);
    auto lhs_abs = this->AbsInt(lhs);
    TRT_ENSURE_OK(lhs_abs);
    auto add1 = this->Add(lhs_abs->output, rhs_abs->output);
    TRT_ENSURE_OK(add1);
    auto one_const = this->Constant<int32>(1, rhs->getDimensions().nbDims);
    TRT_ENSURE_OK(one_const);
    auto numerator = this->Sub(add1->output, one_const->output);
    TRT_ENSURE_OK(numerator);
    return FloorDiv(numerator->output, rhs_abs->output);
  }

  // Adds an operation whose output is equal to "true_values" where the
  // corresponding value in the "bool_condition" tensor is true, otherwise the
  // output has the corresponding value from "false_values". In TensorRT this is
  // called "ISelectLayer", but to conform with Tensorflow naming, we use
  // "where" here.
  StatusOr<CreationResult<nvinfer1::ISelectLayer>> Where(
      nvinfer1::ITensor* bool_condition, nvinfer1::ITensor* true_values,
      nvinfer1::ITensor* false_values) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(bool_condition);
    TRT_EXPECT(true_values);
    TRT_EXPECT(false_values);
    auto layer =
        network_->addSelect(*bool_condition, *true_values, *false_values);
    return MakeResult(layer);
  }

  // Simulates the creation of a tensor of the same shape as "input", but filled
  // with 1's of type int32 where "input" is non-zero, otherwise it is zero.
  // TensorRT doesn't have a "cast" operation, so this simulates with a
  // "select/where" operation.
  StatusOr<CreationResult<nvinfer1::ISelectLayer>> NonZeroInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(input);
    auto zeros = this->Constant<int32>(0, input->getDimensions().nbDims);
    TRT_ENSURE_OK(zeros);
    auto ones = this->Constant<int32>(1, input->getDimensions().nbDims);
    TRT_ENSURE_OK(ones);
    auto equal_to_zero_mask = this->Equal(input, zeros->output);
    TRT_ENSURE_OK(equal_to_zero_mask);
    return this->Where(equal_to_zero_mask->output, input, ones->output);
  }

  // Adds an elementwise multiplication operation to the network.
  StatusOr<CreationResult<nvinfer1::IElementWiseLayer>> Mul(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(lhs);
    TRT_EXPECT(rhs);
    auto layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kPROD);
    return MakeResult(layer);
  }

  // Adds a Constant layer whose output is a TensorRT shape tensor. The shape
  // tensor's size and values correspond to dim's nbDims and d[], respectively.
  StatusOr<CreationResult<nvinfer1::IConstantLayer>> Constant(
      const nvinfer1::Dims& shape_data) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(shape_data.nbDims > 0);
    nvinfer1::Dims shape_dims;
    shape_dims.nbDims = 1;
    shape_dims.d[0] = shape_data.nbDims;
    auto const_weights =
        weight_store_->GetTempWeights(nvinfer1::DataType::kINT32, shape_dims);
    auto values = const_weights.GetPointer<int32>();
    for (int i = 0; i < shape_data.nbDims; i++) {
      values[i] = static_cast<int32>(shape_data.d[i]);
    }
    auto const_layer = network_->addConstant(const_weights.shape_,
                                             const_weights.GetTrtWeights());
    TRT_ENSURE(const_layer);
    auto output = const_layer->getOutput(0);
    TRT_ENSURE(output);
    TRT_ENSURE(output->getType() == nvinfer1::DataType::kINT32);
    return MakeResult(const_layer);
  }

  // Adds a Constant layer composed of a single layer
  template <typename T,
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  StatusOr<CreationResult<nvinfer1::IConstantLayer>> Constant(
      const T scalar, const int nb_dims) noexcept {
    TRT_EXPECT(network_);
    TRT_EXPECT(nb_dims <= nvinfer1::Dims::MAX_DIMS);
    auto data_type = nvinfer1::DataType::kINT32;
    if (std::is_floating_point<T>::value) {
      data_type = nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::Dims zero_shape;
    zero_shape.nbDims = nb_dims;
    std::fill_n(zero_shape.d, nb_dims, 1);
    auto const_weights = weight_store_->GetTempWeights(data_type, zero_shape);
    const_weights.GetPointer<T>()[0] = scalar;
    auto const_layer =
        network_->addConstant(zero_shape, const_weights.GetTrtWeights());
    return MakeResult(const_layer);
  };

  // Adds a TensorRT Slice operation to the network.
  StatusOr<CreationResult<nvinfer1::ISliceLayer>> Slice(
      nvinfer1::ITensor* input, const nvinfer1::Dims& begin,
      const nvinfer1::Dims& size, const nvinfer1::Dims& stride) noexcept {
    TRT_EXPECT(network_);
    nvinfer1::ISliceLayer* layer =
        network_->addSlice(*input, begin, size, stride);
    return MakeResult(layer);
  }

  // Adds a TensorRT Shape operation, which determines the runtime shape of the
  // input tensor, to the network.
  StatusOr<CreationResult<nvinfer1::IShapeLayer>> Shape(
      nvinfer1::ITensor* input) {
    TRT_EXPECT(input);
    TRT_EXPECT(network_);
    return MakeResult(network_->addShape(*input));
  }

  // Creates a Gather operation on the shape of the input tensor. The output of
  // the gather operation is a shape tensor of the same size as the input's
  // shape tensor. The output consists of all zeros except at the specified
  // indices, where it is equal to the given input tensor's shape in the
  // corresponding dimension.
  StatusOr<CreationResult<nvinfer1::IGatherLayer>> GetPartialShapeOf(
      nvinfer1::ITensor* input, absl::InlinedVector<int64, 4> indices) {
    TRT_EXPECT(input);
    TRT_EXPECT(network_);
    TRT_EXPECT(indices.size() <= nvinfer1::Dims::MAX_DIMS);

    // Get the runtime shape of input;
    auto shape_layer = this->Shape(input);
    TRT_ENSURE_OK(shape_layer);
    auto runtime_shape = shape_layer->output;

    // Create a constant tensor containing the gather indices.
    // For any dim not in "indices", we mark it size to gather a zero.
    const auto input_nb_dims = input->getDimensions().nbDims;
    nvinfer1::Dims indices_as_dims;
    indices_as_dims.nbDims = input_nb_dims;
    std::fill_n(indices_as_dims.d, input_nb_dims, input_nb_dims);
    for (auto idx : indices) {
      TRT_EXPECT(idx < input_nb_dims);
      indices_as_dims.d[idx] = idx;
    }
    auto indices_result = this->Constant(indices_as_dims);
    TRT_ENSURE_OK(indices_result);
    auto gather_indices = indices_result->output;
    TRT_ENSURE(gather_indices->getDimensions().nbDims == 1);
    TRT_ENSURE(gather_indices->getType() == nvinfer1::DataType::kINT32);

    // Append a zero to the shape tensor.
    nvinfer1::Dims zero = {1, {0}};
    auto zero_result = this->Constant(zero);
    TRT_ENSURE_OK(zero_result);
    std::array<nvinfer1::ITensor*, 2> cat_inputs = {runtime_shape,
                                                    zero_result->output};
    auto cat_layer =
        network_->addConcatenation(cat_inputs.data(), cat_inputs.size());
    TRT_ENSURE(cat_layer);
    auto gather_input = cat_layer->getOutput(0);
    TRT_ENSURE(gather_input);

    // Finally, gather the indices from
    auto gather = network_->addGather(*gather_input, *gather_indices, 0);
    TRT_ENSURE(gather);
    return MakeResult(gather);
  };

 private:
  // Creates a CreationResult<T>. This helper can use type deduction to simplify
  // the act of returning a CreationResult<T>.
  template <typename T>
  StatusOr<CreationResult<T>> MakeResult(T* layer) {
    TRT_EXPECT(layer);
    auto tensor = layer->getOutput(0);
    TRT_EXPECT(tensor);
    return CreationResult<T>{layer, tensor};
  }

 private:
  nvinfer1::INetworkDefinition* network_;
  TrtWeightStore* weight_store_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
