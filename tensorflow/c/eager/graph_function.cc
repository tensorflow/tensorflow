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
#include "tensorflow/c/eager/graph_function.h"

#include <utility>

#include "tensorflow/c/eager/abstract_function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tracing {
namespace graph {
GraphFunction::GraphFunction(FunctionDef fdef)
    : AbstractFunction(kGraph),
      func_record_(new FunctionRecord(std::move(fdef), {}, true)) {}
GraphFunction::~GraphFunction() {}
absl::Status GraphFunction::GetFunctionDef(const FunctionDef **fdef) {
  *fdef = &(func_record_->fdef());
  return absl::OkStatus();
}
}  // namespace graph
}  // namespace tracing
}  // namespace tensorflow
