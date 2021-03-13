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

#ifndef TENSORFLOW_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_
#define TENSORFLOW_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/grappler/costs/op_performance_data.pb.h"

namespace tensorflow {
namespace grappler {

// A structure to keep the context of op execution, including its shape,
// execution context, and other relevant information.
struct OpContext {
  std::string name;
  std::string device_name;
  OpInfo op_info;
  const FunctionDefLibrary* function_library;  // Not owned.

  OpContext() { function_library = nullptr; }
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_COSTS_OP_CONTEXT_H_
