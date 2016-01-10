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

namespace tensorflow {

typedef FunctionDefHelper FDH;

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

}  // end namespace tensorflow
