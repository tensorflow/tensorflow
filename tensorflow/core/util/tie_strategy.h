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

#ifndef TENSORFLOW_UTIL_TIE_STRATEGY_H_
#define TENSORFLOW_UTIL_TIE_STRATEGY_H_

// This file contains helper routines to deal with tie in various ops and
// kernels.

#include <string>

#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// TieStrategy: the strategy to deal with ties.
// supported:
//   SAMPLE: Sample from the tied values to enforce the constant k size(default).
//   INCLUDE: Include the tied values. If multiple classes have the same prediction value
//            and straddle the top-k boundary, all of those classes are considered to be in the top k.
//   EXCLUDE: Exclude the tied values. If multiple classes have the same prediction value
//            and straddle the top-k boundary, all of those classes are considered to be out of the top k.
enum TieStrategy {
    SAMPLE  = 1,  // Sample from the tied values to enforce the constant k size(default).
    INCLUDE = 2,  // Include the tied values.
    EXCLUDE = 3,  // Exclude the tied values.
};

// Return the string containing the list of valid tie strategies, that can be
// used as an Attr() in REGISTER_OP.
string GetTieStrategyAttrString();

// Forward declaration to avoid including core/framework/graph.proto.
class NodeDef;

// Specialization to parse an attribute directly into a TieStrategy enum.
Status GetNodeAttr(const NodeDef& node_def, StringPiece attr_name,
                   TieStrategy* value);

}  // end namespace tensorflow

#endif  // TENSORFLOW_UTIL_TIE_STRATEGY_H_
