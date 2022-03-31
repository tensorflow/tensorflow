/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TRANSFORMS_PASS_REGISTRATION_H_
#define TENSORFLOW_CORE_TRANSFORMS_PASS_REGISTRATION_H_

#include <memory>

#include "tensorflow/core/transforms/cf_sink/cf_sink.h"
#include "tensorflow/core/transforms/consolidate_attrs/pass.h"
#include "tensorflow/core/transforms/const_dedupe_hoist/pass.h"
#include "tensorflow/core/transforms/drop_unregistered_attribute/output_shapes.h"
#include "tensorflow/core/transforms/graph_to_func/graph_to_func_pass.h"
#include "tensorflow/core/transforms/remapper/remapper_pass.h"
#include "tensorflow/core/transforms/toposort/toposort_pass.h"

namespace mlir {
namespace tfg {

// Generate the code for registering passes for command-line parsing.
#define GEN_PASS_REGISTRATION
#include "tensorflow/core/transforms/passes.h.inc"

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_TRANSFORMS_PASS_REGISTRATION_H_
