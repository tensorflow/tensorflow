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
#ifndef TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_SELECTOR_H_
#define TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_SELECTOR_H_

#include "tensorflow/core/platform/status.h"
#include "tfrt/support/forward_decls.h"  // from @tf_runtime

namespace tensorflow {

class ImmediateExecutionOperation;
class EagerContext;
class AttrBuilder;
class NodeDef;
}  // namespace tensorflow

namespace tfrt {
class OpHandler;
class CoreRuntime;
class Device;

namespace tf {

using ::tensorflow::EagerContext;
using ::tensorflow::ImmediateExecutionOperation;
using ::tensorflow::NodeDef;
using ::tensorflow::Status;

// A helper class to select op handler in op-by-op execution.
class EagerOpHandlerSelector final {
 public:
  EagerOpHandlerSelector(CoreRuntime* core_runtime, EagerContext* eager_context,
                         OpHandler* fallback_op_handler,
                         bool pin_small_ops_to_cpu);
  ~EagerOpHandlerSelector();

  // Selects the op handler to execute the op based on the arguments. This
  // op handler selection is cheap. But it can be nullptr even it return OK
  // status.
  Status SelectFromArguments(const ImmediateExecutionOperation& op,
                             OpHandler** op_handler);

  // Selects the op handler to execute the op based on NodeDef. This op handler
  // selection is expensive. It will never return nullptr unless there is an
  // error. Please only invoke this method when the cheap version fails.
  Status SelectFromNodeDef(const ImmediateExecutionOperation& op,
                           const NodeDef* ndef, OpHandler** op_handler);

 private:
  CoreRuntime* core_runtime_;
  EagerContext* eager_context_;

  const Device& cpu_device_;
  OpHandler* cpu_op_handler_;
  OpHandler* fallback_op_handler_;
  bool pin_small_ops_to_cpu_;
};

}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_CORE_RUNTIME_OP_HANDLER_SELECTOR_H_
