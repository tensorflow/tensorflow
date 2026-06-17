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

#ifndef TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_
#define TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_

#include <complex>

// clang-format off
// This include order is required to avoid instantiating templates
// quantized types in the Eigen namespace before their specialization.
#include "xla/tsl/framework/numeric_types.h"
#include "tensorflow/core/platform/types.h"
// clang-format on

namespace tensorflow {

// NOLINTBEGIN(misc-unused-using-decls)
using tsl::complex128;
using tsl::complex64;

// We use Eigen's QInt implementations for our quantized int types.
using tsl::qint16;
using tsl::qint32;
using tsl::qint8;
using tsl::quint16;
using tsl::quint8;
// NOLINTEND(misc-unused-using-decls)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_NUMERIC_TYPES_H_
