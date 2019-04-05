/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/gl/kernels/registry.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/add.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/concat.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/conv.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/depthwise_conv.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/elementwise.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/fully_connected.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/lstm.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/mul.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/pad.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/pooling.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/prelu.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/relu.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/reshape.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/slice.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/softmax.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/transpose_conv.h"
#include "tensorflow/lite/delegates/gpu/gl/kernels/upsampling_bilinear.h"

#ifndef TFLITE_GPU_BINARY_RELEASE
#include "tensorflow/lite/delegates/gpu/gl/kernels/max_unpooling.h"
#endif  // TFLITE_GPU_BINARY_RELEASE

namespace tflite {
namespace gpu {
namespace gl {
namespace {

class Registry : public NodeShader {
 public:
  Registry() {
    using Type = OperationType;
    using NewShaderFunc = std::function<std::unique_ptr<NodeShader>()>;

    auto insert_op = [&](Type type, NewShaderFunc func) {
      shaders_[ToString(type)].push_back(func());
    };
    auto insert_elementwise_op = [&](Type operation_type) {
      shaders_[ToString(operation_type)].push_back(
          NewElementwiseNodeShader(operation_type));
    };

    insert_op(Type::ADD, NewAddNodeShader);
    insert_op(Type::APPLY_MASK, NewApplyMaskNodeShader);
    insert_op(Type::CONCAT, NewAlignedConcatNodeShader);
    insert_op(Type::CONCAT, NewFlatConcatNodeShader);
    insert_op(Type::CONCAT, NewConcatNodeShader);
    insert_op(Type::CONVOLUTION_2D, NewConvolution1x1NodeShader);
    insert_op(Type::CONVOLUTION_2D, NewConvolutionNodeShader);
    insert_op(Type::CONVOLUTION_TRANSPOSED, NewConvolutionTransposedNodeShader);
    insert_op(Type::DEPTHWISE_CONVOLUTION, NewDepthwiseConvolutionNodeShader);
    insert_op(Type::FULLY_CONNECTED, NewFullyConnectedNodeShader);
    insert_op(Type::LSTM, NewLstmNodeShader);
    insert_op(Type::MULTIPLY_SCALAR, NewMultiplyScalarNodeShader);
    insert_op(Type::PAD, NewPadNodeShader);
    insert_op(Type::POOLING_2D, NewPoolingNodeShader);
    insert_op(Type::RELU, NewReLUNodeShader);
    insert_op(Type::RESHAPE, NewReshapeNodeShader);
    insert_op(Type::PRELU, NewPReLUNodeShader);
    insert_op(Type::SLICE, NewSliceNodeShader);
    insert_op(Type::SOFT_MAX, NewSoftMaxNodeShader);
    insert_op(Type::UPSAMPLE_2D, NewUpsamplingNodeShader);

    insert_elementwise_op(Type::ABS);
    insert_elementwise_op(Type::COS);
    insert_elementwise_op(Type::LOG);
    insert_elementwise_op(Type::RSQRT);
    insert_elementwise_op(Type::SIGMOID);
    insert_elementwise_op(Type::SIN);
    insert_elementwise_op(Type::SQRT);
    insert_elementwise_op(Type::SQUARE);
    insert_elementwise_op(Type::TANH);
    insert_elementwise_op(Type::SUB);
    insert_elementwise_op(Type::DIV);
    insert_elementwise_op(Type::POW);
    insert_elementwise_op(Type::SQUARED_DIFF);

#ifndef TFLITE_GPU_BINARY_RELEASE
    insert_op(Type::MAX_UNPOOLING_2D, NewMaxUnpoolingNodeShader);
#endif  // TFLITE_GPU_BINARY_RELEASE
  }

  ~Registry() final = default;

  Status GenerateCode(const GenerationContext& ctx,
                      GeneratedCode* generated_code) const final {
    std::vector<std::string> errors;
    auto it = shaders_.find(ctx.node->operation.type);
    if (it != shaders_.end()) {
      for (auto& shader : it->second) {
        const auto status = shader->GenerateCode(ctx, generated_code);
        if (status.ok()) return status;
        errors.push_back(status.error_message());
      }
    }
    return NotFoundError(absl::StrCat("Suitable node shader is not found: ",
                                      absl::StrJoin(errors, ", ")));
  }

 private:
  std::unordered_map<std::string, std::vector<std::unique_ptr<NodeShader>>>
      shaders_;
};

}  // namespace

std::unique_ptr<NodeShader> NewNodeShaderRegistry() {
  return absl::make_unique<Registry>();
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
