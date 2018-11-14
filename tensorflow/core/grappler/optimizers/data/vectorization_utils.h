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

// Given a MapDefun node (`map_defun_node`) in a FunctionDef (`outer_scope`)
// that maps a function in lib across some input vector elements,
// `VectorizeMapDefun` attempts to create a vectorized version of `outer_scope`
// by "lifting" operations from the MapDefun function to the new function
// (`result`); that is, replacing operations in the MapDefun function with
// operations that produce the same vector output(s) as executing the original
// operations on elements of vector input(s) would. If all operations in the
// MapDefun function are successfully lifted, `result` has no MapDefun node
// altogether. However, if some operations cannot be lifted, and this
// vectorization only succeeds partially, a MapDefun node remains in `result` to
// be used for operations that were not lifted, and the modified MapDefun
// function is added to `lib`. The newly vectorized function `result` is also
// added to `lib`.
//
// Returns Status::OK() if the vectorization is completely or partially
// successful. Otherwise, returns an error, and sets `result` to nullptr.
//
// Example:
//   If the input to the `VectorizeMapDefun` function is a MapDefun
// whose `map_defun_fn` performs the Cast operation, the vectorization will
// eliminate the MapDefun. This is because the Cast operation supports
// any tensor shape and can thus be lifted to `result`.
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
// result          +------+
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
Status VectorizeMapDefun(const FunctionDef& outer_scope,
                         const NodeDef& map_defun_node, FunctionDefLibrary* lib,
                         FunctionDef** result);

}  // end namespace vectorization_utils
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_VECTORIZATION_UTILS_H_
