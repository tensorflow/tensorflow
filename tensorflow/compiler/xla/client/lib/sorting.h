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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SORTING_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SORTING_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Returns a tuple composed of the top `k` values and corresponding indices in
// `input`.  Output values are in descending order, from largest to smallest.
XlaOp TopK(XlaOp input, int64 k);
// Split sort in TopK into smaller sorts.
// Returns a tuple composed of the top `k` values and corresponding indices in
// `input`.  Output values are in descending order, from largest to smallest.
XlaOp TopKWithPartitions(XlaOp input, int64 k, int64 num_partitions = 1);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_SORTING_H_
