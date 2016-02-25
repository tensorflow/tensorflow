/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

typedef FunctionDefHelper FDH;

Status SoftmaxGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      "SoftmaxGrad",
      // Arg defs
      {"x: T", "grad_softmax: T"},
      // Ret val defs
      {"grad_x: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      // Based on _SoftmaxGrad in nn_grad.py.
      {
        {{"softmax"}, "Softmax", {"x"}, {{"T", "$T"}}},
        {{"n0"}, "Mul", {"grad_softmax", "softmax"}, {{"T", "$T"}}},
        FDH::Const<int32>("indices", {1}),
        {{"n1"}, "Sum", {"n0", "indices"}, {{"T", "$T"}}},
        FDH::Const<int32>("newshape", {-1, 1}),
        {{"n2"}, "Reshape", {"n1", "newshape"}, {{"T", "$T"}}},
        {{"n3"}, "Sub", {"grad_softmax", "n2"}, {{"T", "$T"}}},
        {{"grad_x"}, "Mul", {"n3", "softmax"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Softmax", SoftmaxGrad);

Status ReluGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "dy: T"},
      // Ret val defs
      {"dx: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
        {{"dx"}, "ReluGrad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Relu", ReluGrad);

Status CrossEntropyGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"features: T", "labels: T", "dcost_dloss: T", "donotcare: T"},
    // Ret val defs
    {"dcost_dfeatures: T", "dcost_dlabels: T"},
    // Attr defs
    {{"T: {float, double}"}},
    // Nodes
    {
      // _, dloss_dfeatures = CrossEntropy(features, labels)
      {{"donotcare_loss", "dloss_dfeatures"}, "CrossEntropy",
       {"features", "labels"}, {{"T", "$T"}}},
      // dcost_dloss is of shape [batch_size].
      // dcost_dloss_mat is of shape [batch_size, 1].
      FDH::Const("neg1", -1),
      {{"dcost_dloss_mat"}, "ExpandDims", {"dcost_dloss", "neg1"},
       {{"T", "$T"}}},
      // chain rule: dcost/dfeatures = dcost/dloss * dloss/dfeatures
      {{"dcost_dfeatures"}, "Mul", {"dcost_dloss_mat", "dloss_dfeatures"},
       {{"T", "$T"}}},
      {{"dcost_dlabels"}, "ZerosLike", {"labels"}, {{"T", "$T"}}},
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("CrossEntropy", CrossEntropyGrad);

Status Conv2DGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "filter: T", "grad: T"},
    // Ret val defs
    {"input_grad: T", "filter_grad: T"},
    // Attr defs
    {"T: {float, double}",
     "strides: list(int)",
     "use_cudnn_on_gpu: bool = true",
     GetPaddingAttrString(),
     GetConvnetDataFormatAttrString()},
    // Nodes
    {
      {{"i_shape"}, "Shape", {"input"}, {{"T", "$T"}}},
      {{"input_grad"}, "Conv2DBackpropInput", {"i_shape", "filter", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"strides", "$strides"},
                  {"padding", "$padding"},
                  {"data_format", "$data_format"},
                  {"use_cudnn_on_gpu", "$use_cudnn_on_gpu"}}},

      {{"f_shape"}, "Shape", {"filter"}, {{"T", "$T"}}},
      {{"filter_grad"}, "Conv2DBackpropFilter", {"input", "f_shape", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"strides", "$strides"},
                  {"padding", "$padding"},
                  {"data_format", "$data_format"},
                  {"use_cudnn_on_gpu", "$use_cudnn_on_gpu"}}},
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Conv2D", Conv2DGrad);

Status MaxPoolGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: float", "grad: float"},
    // Ret val defs
    {"output: float"},
    // Attr defs
    {"ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPool", MaxPoolGrad);

}  // end namespace tensorflow
