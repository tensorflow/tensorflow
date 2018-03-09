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

#ifndef TENSORFLOW_COMPILER_XLA_TYPES_H_
#define TENSORFLOW_COMPILER_XLA_TYPES_H_

#include <complex>

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"

#include <Eigen/Core>

namespace xla {

using ::tensorflow::string;

using ::tensorflow::int8;
using ::tensorflow::int16;
using ::tensorflow::int32;
using ::tensorflow::int64;

using ::tensorflow::bfloat16;

using ::tensorflow::uint8;
using ::tensorflow::uint16;
using ::tensorflow::uint32;
using ::tensorflow::uint64;

using complex64 = std::complex<float>;

using ::Eigen::half;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TYPES_H_
