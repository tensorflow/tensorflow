/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPPLER_OP_TYPES_H_
#define TENSORFLOW_GRAPPLER_OP_TYPES_H_

#include "tensorflow/core/framework/node_def.pb.h"

namespace tensorflow {
namespace grappler {

bool IsConcat(const NodeDef& node);
bool IsConstant(const NodeDef& node);
bool IsDequeueOp(const NodeDef& node);
bool IsIdentity(const NodeDef& node);
bool IsMerge(const NodeDef& node);
bool IsNextIteration(const NodeDef& node);
bool IsNoOp(const NodeDef& node);
bool IsPlaceholder(const NodeDef& node);
bool IsRecv(const NodeDef& node);
bool IsReduction(const NodeDef& node);
bool IsReshape(const NodeDef& node);
bool IsSend(const NodeDef& node);
bool IsSwitch(const NodeDef& node);
bool IsTranspose(const NodeDef& node);
bool IsVariable(const NodeDef& node);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OP_TYPES_H_
