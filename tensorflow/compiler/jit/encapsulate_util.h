/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// This file contains some utility functions for encapsulating XLA computation
// in host graph and encapsulating outside compilation in XLA computation.

#ifndef TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_

#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// Attribute marking output tensor shapes inferred by XLA. Attribute value is
// a list of PartialTensorShape objects.
extern const char kXlaInferredShapesAttrName[];

// Infer output shapes for outside compilation nodes which have output data
// edges to XLA computation nodes. These shapes will be used later by XLA
// compiler as output shapes of the outside compilation's XlaHostCompute op.
// XLA computation nodes will be mark by attr `xla_computation_attr_name`;
// outside compilation nodes will be marked by both attr
// `xla_computation_attr_name` and `outside_compilation_attr_name`.
//
// Those outside compilation nodes will be marked with attribute
// `kXlaInferredShapesAttrName`.
//
// We have to perform shape inference before encapsulation because after
// encapsulation, some nodes will be encapsulated into function call, and shape
// inference does not handle function call at the moment.
Status PerformStaticShapeInferenceBeforeEncapsulation(
    Graph* g, const string& xla_computation_attr_name,
    const string& outside_compilation_attr_name);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_UTIL_H_
