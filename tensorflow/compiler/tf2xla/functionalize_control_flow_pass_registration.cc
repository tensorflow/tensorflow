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

#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"

namespace tensorflow {

// This pass is required for some AOT backends and all JIT backends, so this
// file exists as a separate lib and will be linked to both AOT and JIT.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 27,
                      FunctionalizeControlFlowPass);

}  // namespace tensorflow
