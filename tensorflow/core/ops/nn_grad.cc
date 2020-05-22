/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#if defined(INTEL_MKL)
      {{"T: {float, double, bfloat16}"}},
#else
      {{"T: {float, double}"}},
#endif
      // Nodes
      // Based on _SoftmaxGrad in nn_grad.py.
      {
        {{"softmax"}, "Softmax", {"x"}, {{"T", "$T"}}},
        {{"n0"}, "Mul", {"grad_softmax", "softmax"}, {{"T", "$T"}}},
        FDH::Const<int32>("indices", {-1}),
        {{"n1"}, "Sum", {"n0", "indices"}, {{"keep_dims", true}, {"T", "$T"}}},
        {{"n2"}, "Sub", {"grad_softmax", "n1"}, {{"T", "$T"}}},
        {{"grad_x"}, "Mul", {"n2", "softmax"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Softmax", SoftmaxGrad);

Status LogSoftmaxGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      "LogSoftmaxGrad",
      // Arg defs
      {"x: T", "grad_logsoftmax: T"},
      // Ret val defs
      {"grad_x: T"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      // Based on _LogSoftmaxGrad in nn_grad.py.
      {
        {{"softmax"}, "Softmax", {"x"}, {{"T", "$T"}}},
        FDH::Const<int32>("indices", {-1}),
        {{"n0"}, "Sum", {"grad_logsoftmax", "indices"},
         {{"keep_dims", true}, {"T", "$T"}}},
        {{"n1"}, "Mul", {"n0", "softmax"}, {{"T", "$T"}}},
        {{"grad_x"}, "Sub", {"grad_logsoftmax", "n1"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("LogSoftmax", LogSoftmaxGrad);

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

Status Relu6Grad(const AttrSlice& attrs, FunctionDef* g) {
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
        {{"dx"}, "Relu6Grad", {"dy", "x"}, {{"T", "$T"}}}
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Relu6", Relu6Grad);

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
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPool", MaxPoolGrad);

Status AvgPoolGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      {{"i_shape"}, "Shape", {"input"}, {{"T", "$T"}}},
      {{"output"}, "AvgPoolGrad", {"i_shape", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("AvgPool", AvgPoolGrad);

Status MaxPoolGradGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "grad: T"},
    // Ret val defs
    {"output: T"},
    // Attr defs
    {"T: {float, half} = DT_FLOAT",
     "ksize: list(int) >= 4",
     "strides: list(int) >= 4",
     GetPaddingAttrString()},
    // Nodes
    {
      // Invoke MaxPool again to recompute the outputs (removed by CSE?).
      {{"maxpool"}, "MaxPool", {"input"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}},
      {{"output"}, "MaxPoolGradGrad", {"input", "maxpool", "grad"},
       /*Attrs=*/{{"T", "$T"},
                  {"ksize", "$ksize"},
                  {"strides", "$strides"},
                  {"padding", "$padding"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("MaxPoolGrad", MaxPoolGradGrad);

Status BiasAddGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
    // Arg defs
    {"input: T", "bias: T", "grad: T"},
    // Ret val defs
    {"grad: T", "bias_grad: T"},
    // Attr defs
    {{"T: {float, double}"},
     GetConvnetDataFormatAttrString()},
    // Nodes
    {
      {{"bias_grad"}, "BiasAddGrad", {"grad"},
           /*Attrs=*/{{"T", "$T"},
                      {"data_format", "$data_format"}}}
    });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("BiasAdd", BiasAddGrad);

}  // end namespace tensorflow
