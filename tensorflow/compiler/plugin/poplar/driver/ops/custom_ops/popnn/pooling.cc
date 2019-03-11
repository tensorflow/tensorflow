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

#include "tensorflow/compiler/plugin/poplar/driver/ops/custom_ops/poplibs_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"

#include <popnn/Pooling.hpp>
#include <popnn/PoolingDef.hpp>

#include <string>

namespace xla {
namespace poplarplugin {
namespace {
class MaxPoolOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    TF_ASSIGN_OR_RETURN(Window window,
                        attribute_map.GetAttributeAsWindow("window"));
    return CreatePoplibsPooling(res, inst, tensor_map, popnn::PoolingType::MAX,
                                window);
  }
};
REGISTER_POPLIBS_OP(Popnn, MaxPool, MaxPoolOp);

class AvgPoolOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    TF_ASSIGN_OR_RETURN(Window window,
                        attribute_map.GetAttributeAsWindow("window"));
    return CreatePoplibsPooling(res, inst, tensor_map, popnn::PoolingType::AVG,
                                window);
  }
};
REGISTER_POPLIBS_OP(Popnn, AvgPool, AvgPoolOp);

class MaxPoolGradOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    TF_ASSIGN_OR_RETURN(Window window,
                        attribute_map.GetAttributeAsWindow("window"));
    return CreatePoplibsMaxPoolGrad(res, inst, tensor_map, window);
  }
};
REGISTER_POPLIBS_OP(Popnn, MaxPoolGrad, MaxPoolGradOp);

class AvgPoolGradOp : public PoplibsOpDef {
  StatusOr<poplar::program::Program> Creator(
      poplar::Graph& graph, CompilerResources& res, const HloInstruction* inst,
      const xla::Shape& output_shape, TensorMap& tensor_map,
      const IPUCustomKernelsUtil::AttributeMap& attribute_map) override {
    TF_ASSIGN_OR_RETURN(Window window,
                        attribute_map.GetAttributeAsWindow("window"));
    return CreatePoplibsPoolingGrad(res, inst, tensor_map,
                                    popnn::PoolingType::AVG, window);
  }
};
REGISTER_POPLIBS_OP(Popnn, AvgPoolGrad, AvgPoolGradOp);

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
