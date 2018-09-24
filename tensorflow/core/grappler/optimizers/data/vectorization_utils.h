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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_UTILS_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_UTILS_H_

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace grappler {
namespace vectorization_utils {

// Given a function, `map_defun_fn`, that is mapped across some input vector
// elements via a MapDefun operation, `VectorizeMapDefun` attempts to
// vectorize the MapDefun by "lifting" operations from the `map_defun_fn` to the
// `outer_scope`; that is, replacing `map_defun_fn` operations with new
// `outer_scope` operations that produce the same vector output(s) as executing
// the `map_defun_fn` operations on elements of vector input(s) would. If all
// `map_defun_fn` operations are successfully lifted, `map_defun_node` is
// eliminated from `outer_scope` altogether. However, if some operations cannot
// be lifted, and this vectorization only succeeds partially, `map_defun_node`
// remains to be used for operations that were not lifted.
//
// Example:
//   If the input to the `VectorizeMapDefun` function is a MapDefun
// whose `map_defun_fn` performs the Cast operation, the vectorization will
// eliminate the MapDefun. This is because the Cast operation supports
// any tensor shape and can thus be lifted to the `outer_scope`.
//
// Before:
//
//
// outer_scope     +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |  map_defun_fn +---v--+         |
// |   +-----------+ Arg0 +-----+   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   |               |        |   |
// |   |           +---v--+     |   |
// |   |           | Cast |     |   |
// |   |           +---+--+     |   |
// |   |               |        |   |
// |   |           +---v--+     |   |
// |   +-----------+ Ret0 +-----+   |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
//
// After:
//
// outer_scope     +------+
// +---------------+ Arg0 +---------+
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// |               | Cast |         |
// |               +---+--+         |
// |                   |            |
// |               +---v--+         |
// +---------------+ Ret0 +---------+
//                 +------+
//
void VectorizeMapDefun(FunctionDef* outer_scope, FunctionDef* map_defun_fn,
                       NodeDef* map_defun_node);

}  // end namespace vectorization_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_UTILS_H_
