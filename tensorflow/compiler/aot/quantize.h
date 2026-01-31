/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_AOT_QUANTIZE_H_
#define TENSORFLOW_COMPILER_AOT_QUANTIZE_H_

#include <functional>
#include <iostream>
#include <ostream>

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "xla/hlo/builder/xla_computation.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tfcompile {

using QuantizeXlaFn = std::function<absl::Status(
    const tf2xla::Config& config, xla::XlaComputation* computation)>;

// Set the static quantization function to the `fn` if it hasn't been set.
// Return false if the static function has been set.
bool RegisterQuantizeFn(const QuantizeXlaFn& fn);

}  // namespace tfcompile
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_AOT_QUANTIZE_H_
