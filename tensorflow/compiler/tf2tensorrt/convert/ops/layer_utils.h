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

// Facilitates the creation of TensorRT layers inside a network. The user
// provides a INetworkDefinition pointer during construction. They can then add
// operations to the network through the provided functions. Each function
// returns a struct which contains the symbolic result of the operation (ITensor
// pointer) as well as a pointer to the last TensorRT ILayer created. Some
// operations may create multiple layers in order to accomplish the desired
// result (e.g. Sign).
class TRTNetworkBuilder {
 public:
  static StatusOr<TRTNetworkBuilder> Create(
      nvinfer1::INetworkDefinition* network, TrtWeightStore* weight_store) {
    TRT_ENSURE(network);
    TRT_ENSURE(weight_store);
    return TRTNetworkBuilder(network, weight_store);
  }

 private:
  TRTNetworkBuilder(nvinfer1::INetworkDefinition* network,
                    TrtWeightStore* weight_store)
      : network_(network), weight_store_(weight_store) {}

 public:
  // Adds an Add operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Add(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kSUM);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an elementwise min(lhs, rhs) operation to the network. The output has
  // the same data type as the input.
  StatusOr<nvinfer1::IElementWiseLayer*> Min(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kMIN);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an elementwise max(lhs, rhs) operation to the network. The output has
  // the same datatype as the input.
  StatusOr<nvinfer1::IElementWiseLayer*> Max(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kMAX);
    TRT_ENSURE(layer);
    return layer;
  };

  // Adds an absolute value operation to the network. Note that this unary
  // operation will do an implict float conversion. For int32 tensors, use
  // "AbsInt".
  StatusOr<nvinfer1::IUnaryLayer*> AbsFloat(nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);
    TRT_ENSURE(input->getType() != nvinfer1::DataType::kFLOAT &&
               input->getType() != nvinfer1::DataType::kHALF);
    nvinfer1::IUnaryLayer* layer =
        network_->addUnary(*input, nvinfer1::UnaryOperation::kABS);
    TRT_ENSURE(layer);
    return layer;
  }

  // Performs Abs without implict float conversion. The input should be of type
  // kInt32. For float datatypes, use "Abs".
  StatusOr<nvinfer1::IElementWiseLayer*> AbsInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);
    TRT_ENSURE(input->getType() == nvinfer1::DataType::kINT32);
    StatusOr<nvinfer1::IElementWiseLayer*> sign = this->SignInt(input);
    return this->Mul(input, (*sign)->getOutput(0));
  }

  // Returns elementwise sign(x) for int32 input tensors where sign(x) is
  // defined as 1 where x > 0, -1 where x < 0 and 0 where x == 0.
  StatusOr<nvinfer1::IElementWiseLayer*> SignInt(
      nvinfer1::ITensor* input) noexcept {
    TRT_ENSURE(input);

    // Create constants +1 and -1.
    StatusOr<nvinfer1::IConstantLayer*> one =
        this->Constant<int32>(1, input->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(one);

    StatusOr<nvinfer1::IConstantLayer*> neg_one =
        this->Constant<int32>(-1, input->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(neg_one);

    // Turn all negaitve elements into -1, positive and zero elements
    // unaffected.
    StatusOr<nvinfer1::IElementWiseLayer*> max =
        this->Max(input, (*neg_one)->getOutput(0));
    TRT_ENSURE_PTR_OK(max);

    // Turn all positive elements into +1, negative and zero elements
    // unaffected.
    StatusOr<nvinfer1::IElementWiseLayer*> min =
        this->Min((*max)->getOutput(0), (*one)->getOutput(0));
    TRT_ENSURE_PTR_OK(min);
    return min;
  }

  // Adds a Sub operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Sub(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kSUB);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds an Greater operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Greater(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kGREATER);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds an Equal operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Equal(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kEQUAL);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a FloorDiv operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> FloorDiv(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kFLOOR_DIV);
    TRT_ENSURE(layer);
    return layer;
  }

  // Returns the equivalent of ceil_divide(abs(x)/abs(y))) operation. The inputs
  // "lhs" and "rhs" should be int32 tensors.
  StatusOr<nvinfer1::IElementWiseLayer*> AbsCeilDivInt(
      nvinfer1::ITensor* lhs, nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    TRT_ENSURE(lhs->getType() == nvinfer1::DataType::kINT32);
    TRT_ENSURE(rhs->getType() == nvinfer1::DataType::kINT32);

    StatusOr<nvinfer1::IElementWiseLayer*> rhs_abs = this->AbsInt(rhs);
    TRT_ENSURE_PTR_OK(rhs_abs);
    StatusOr<nvinfer1::IElementWiseLayer*> lhs_abs = this->AbsInt(lhs);
    TRT_ENSURE_PTR_OK(lhs_abs);
    StatusOr<nvinfer1::IElementWiseLayer*> add1 =
        this->Add((*lhs_abs)->getOutput(0), (*rhs_abs)->getOutput(0));
    TRT_ENSURE_PTR_OK(add1);
    StatusOr<nvinfer1::IConstantLayer*> one_const =
        this->Constant<int32>(1, rhs->getDimensions().nbDims);
    TRT_ENSURE_PTR_OK(one_const);
    StatusOr<nvinfer1::IElementWiseLayer*> numerator =
        this->Sub((*add1)->getOutput(0), (*one_const)->getOutput(0));
    TRT_ENSURE_PTR_OK(numerator);
    return FloorDiv((*numerator)->getOutput(0), (*rhs_abs)->getOutput(0));
  }

  // Adds an elementwise multiplication operation to the network.
  StatusOr<nvinfer1::IElementWiseLayer*> Mul(nvinfer1::ITensor* lhs,
                                             nvinfer1::ITensor* rhs) noexcept {
    TRT_ENSURE(lhs);
    TRT_ENSURE(rhs);
    nvinfer1::IElementWiseLayer* layer = network_->addElementWise(
        *lhs, *rhs, nvinfer1::ElementWiseOperation::kPROD);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a Constant layer whose output is a TensorRT shape tensor. The shape
  // tensor's size and values correspond to dim's nbDims and d[], respectively.
  StatusOr<nvinfer1::IConstantLayer*> ConstantShape(
      const nvinfer1::Dims& shape_data) noexcept {
    TRT_ENSURE(shape_data.nbDims > 0);
    nvinfer1::Dims shape_dims;
    shape_dims.nbDims = 1;
    shape_dims.d[0] = shape_data.nbDims;
    TRT_ShapedWeights const_weights =
        weight_store_->GetTempWeights(nvinfer1::DataType::kINT32, shape_dims);
    int32* values = const_weights.GetPointer<int32>();
    for (int i = 0; i < shape_data.nbDims; i++) {
      values[i] = static_cast<int32>(shape_data.d[i]);
    }
    nvinfer1::IConstantLayer* const_layer = network_->addConstant(
        const_weights.shape_, const_weights.GetTrtWeights());
    TRT_ENSURE(const_layer);
    nvinfer1::ITensor* output = const_layer->getOutput(0);
    TRT_ENSURE(output);
    TRT_ENSURE(output->getType() == nvinfer1::DataType::kINT32);
    TRT_ENSURE(const_layer);
    return const_layer;
  }

  // Adds a Constant layer whose output is a TensorRT shape tensor. The shape
  // tensor's size and values correspond to dim's nbDims and d[], respectively.
  StatusOr<nvinfer1::IConstantLayer*> Constant(
      const std::vector<int>& data) noexcept {
    nvinfer1::Dims shape_dims;
    shape_dims.nbDims = 1;
    shape_dims.d[0] = data.size();
    TRT_ShapedWeights const_weights =
        weight_store_->GetTempWeights(nvinfer1::DataType::kINT32, shape_dims);
    int32* values = const_weights.GetPointer<int32>();
    for (int i = 0; i < data.size(); i++) {
      values[i] = static_cast<int32>(data[i]);
    }
    nvinfer1::IConstantLayer* const_layer = network_->addConstant(
        const_weights.shape_, const_weights.GetTrtWeights());
    TRT_ENSURE(const_layer);
    nvinfer1::ITensor* output = const_layer->getOutput(0);
    TRT_ENSURE(output);
    TRT_ENSURE(output->getType() == nvinfer1::DataType::kINT32);
    TRT_ENSURE(const_layer);
    return const_layer;
  }

  // Adds a Constant layer that produces a tensor with a single value "scalar".
  // The tensor has "nb_dims" dimensions and each dimension has only one
  // element. The data type of the tensor is determined by the data type of
  // "scalar".
  template <typename T,
            typename std::enable_if<std::is_pod<T>::value>::type* = nullptr>
  StatusOr<nvinfer1::IConstantLayer*> Constant(const T scalar,
                                               const int nb_dims) noexcept {
    TRT_ENSURE(nb_dims <= nvinfer1::Dims::MAX_DIMS);
    auto data_type = nvinfer1::DataType::kINT32;
    if (std::is_floating_point<T>::value) {
      data_type = nvinfer1::DataType::kFLOAT;
    }
    nvinfer1::Dims zero_shape;
    zero_shape.nbDims = nb_dims;
    std::fill_n(zero_shape.d, nb_dims, 1);
    TRT_ShapedWeights const_weights =
        weight_store_->GetTempWeights(data_type, zero_shape);
    const_weights.GetPointer<T>()[0] = scalar;
    nvinfer1::IConstantLayer* const_layer =
        network_->addConstant(zero_shape, const_weights.GetTrtWeights());
    TRT_ENSURE(const_layer);
    return const_layer;
  };

  // Adds a TensorRT Slice operation to the network.
  StatusOr<nvinfer1::ISliceLayer*> Slice(
      nvinfer1::ITensor* input, const nvinfer1::Dims& begin,
      const nvinfer1::Dims& size, const nvinfer1::Dims& stride) noexcept {
    nvinfer1::ISliceLayer* layer =
        network_->addSlice(*input, begin, size, stride);
    TRT_ENSURE(layer);
    return layer;
  }

  // Adds a TensorRT Shape operation, which determines the runtime shape of the
  // input tensor, to the network.
  StatusOr<nvinfer1::IShapeLayer*> Shape(nvinfer1::ITensor* input) {
    TRT_ENSURE(input);
    nvinfer1::IShapeLayer* layer = network_->addShape(*input);
    TRT_ENSURE(layer);
    return layer;
  }

  // Creates a Gather operation on the shape of the input tensor. The output of
  // the gather operation is a 1D shape tensor where output[i] = (!sub_one ?
  // input_shape[i] : input_shape[i] -1) if i is in "indices", otherwise zero.
  StatusOr<nvinfer1::IGatherLayer*> GetPartialShapeOf(
      nvinfer1::ITensor* input, absl::InlinedVector<int64, 4> indices,
      bool sub_one = false) {
    TRT_ENSURE(input);
    TRT_ENSURE(indices.size() <= nvinfer1::Dims::MAX_DIMS);

    // Get the runtime shape of input;
    StatusOr<nvinfer1::IShapeLayer*> shape_layer = this->Shape(input);
    TRT_ENSURE_PTR_OK(shape_layer);
    nvinfer1::ITensor* runtime_shape = (*shape_layer)->getOutput(0);

    if (sub_one) {
      StatusOr<nvinfer1::IConstantLayer*> ones = this->Constant<int32>(1, 1);
      TRT_ENSURE_PTR_OK(ones);
      StatusOr<nvinfer1::IElementWiseLayer*> sub =
          this->Sub(runtime_shape, (*ones)->getOutput(0));
      TRT_ENSURE_PTR_OK(sub);
      runtime_shape = (*sub)->getOutput(0);
    }

    // Create a constant tensor containing the gather indices.
    // For any dim not in "indices", we mark it size to gather a zero.
    const int input_nb_dims = input->getDimensions().nbDims;
    std::vector<int> indices_all(input_nb_dims, input_nb_dims);
    for (auto idx : indices) {
      TRT_ENSURE(idx < input_nb_dims);
      indices_all[idx] = idx;
    }

    StatusOr<nvinfer1::IConstantLayer*> indices_result =
        this->Constant(indices_all);
    TRT_ENSURE_PTR_OK(indices_result);
    nvinfer1::ITensor* gather_indices = (*indices_result)->getOutput(0);
    TRT_ENSURE(gather_indices->getDimensions().nbDims == 1);
    TRT_ENSURE(gather_indices->getType() == nvinfer1::DataType::kINT32);

    // Append a zero to the shape tensor.
    StatusOr<nvinfer1::IConstantLayer*> zero_result =
        this->Constant(std::vector<int>{0});
    TRT_ENSURE_PTR_OK(zero_result);
    std::array<nvinfer1::ITensor*, 2> cat_inputs = {
        runtime_shape, (*zero_result)->getOutput(0)};
    nvinfer1::IConcatenationLayer* cat_layer =
        network_->addConcatenation(cat_inputs.data(), cat_inputs.size());
    TRT_ENSURE(cat_layer);
    nvinfer1::ITensor* gather_input = cat_layer->getOutput(0);
    TRT_ENSURE(gather_input);

    // Finally, gather the indices from the input.
    nvinfer1::IGatherLayer* gather =
        network_->addGather(*gather_input, *gather_indices, 0);
    TRT_ENSURE(gather);
    return gather;
  };

 private:
  nvinfer1::INetworkDefinition* const network_;
  TrtWeightStore* const weight_store_;
};

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_OPS_LAYER_UTILS_H_
