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

#ifndef TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_
#define TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Determines if actual and expected are equal, ignoring versions and ordering
// of nodes, attrs, and control inputs.  If the GraphDefs are different and
// diff != nullptr, *diff is set to an explanation of the difference.  Note that
// we use node names to match up nodes between the graphs, and so the naming of
// nodes must be consistent.
bool EqualGraphDef(const GraphDef& actual, const GraphDef& expected,
                   string* diff);

// Determines if actual and expected are equal, ignoring: ordering of
// attrs, internal attributes, and control inputs.
//
// If the NodeDefs are different and
// diff != nullptr, *diff is set to an explanation of the difference.
bool EqualNodeDef(const NodeDef& actual, const NodeDef& expected, string* diff);

#define TF_EXPECT_GRAPH_EQ(expected, actual)                  \
  do {                                                        \
    string diff;                                              \
    EXPECT_TRUE(EqualGraphDef(actual, expected, &diff))       \
        << diff << "\nActual: " << SummarizeGraphDef(actual); \
  } while (false)

}  // namespace tensorflow

#endif  // TENSORFLOW_GRAPH_EQUAL_GRAPH_DEF_H_
