/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
#define TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/random/philox_random.h"

namespace tensorflow {

// 'Variable' doesn't support uint32 or uint64 yet (due to reasons explained
// in b/111604096 and cl/171681867), so we use signed int here. We choose int64
// instead of int32 because `VarHandleOp` doesn't support int32 on GPU, and
// because of the "int32 problem".
using StateElementType = int64;
static constexpr DataType STATE_ELEMENT_DTYPE = DT_INT64;
static constexpr DataType ALGORITHM_DTYPE = STATE_ELEMENT_DTYPE;

using random::PhiloxRandom;

static constexpr int64 PHILOX_MIN_STATE_SIZE =
    (PhiloxRandom::ResultType::kElementCount +
     PhiloxRandom::Key::kElementCount) /
    2;
static constexpr int64 THREEFRY_MIN_STATE_SIZE = 2;

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_STATEFUL_RANDOM_OPS_H_
