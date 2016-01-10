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

REGISTER_OP_NO_GRADIENT("Shape");
REGISTER_OP_NO_GRADIENT("Rank");
REGISTER_OP_NO_GRADIENT("Size");

Status ReshapeGrad(const AttrSlice& attrs, FunctionDef* g) {
  // clang-format off
  *g = FDH::Define(
      // Arg defs
      {"x: T", "shape: int32", "dy: T"},
      // Ret val defs
      {"dx: T", "dshape: int32"},
      // Attr defs
      {{"T: {float, double}"}},
      // Nodes
      {
        {{"x_shape"}, "Shape", {"x"}, {{"T", "$T"}}},
        {{"dx"}, "Reshape", {"dy", "x_shape"}, {{"T", "$T"}}},
        {{"dshape"}, "ZerosLike", {"shape"}, {{"T", DT_INT32}}},
      });
  // clang-format on
  return Status::OK();
}
REGISTER_OP_GRADIENT("Reshape", ReshapeGrad);

}  // end namespace tensorflow
