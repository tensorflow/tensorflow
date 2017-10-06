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

#ifndef TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_H_
#define TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_H_

#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

using FunctionCompiler =
    std::function<xla::StatusOr<std::shared_ptr<xla::Computation>>(
        const NameAttrList& function, XlaOpKernelContext* xla_op_context)>;

// GraphCompiler compiles the graph in topological order in the current
// thread. It also resolves the nondeterminism in the graph by enforcing a total
// order on all inputs to a node. This abstraction helps us create the same XLA
// computation given two structurally equivalent TensorFlow graphs. If a
// function call is visited during the graph traversal, it is then compiled
// through the FunctionCompiler into a computation and a `Call` operation is
// inserted to call into that computation.
class GraphCompiler {
 public:
  GraphCompiler(XlaContext* xla_context, XlaCompilationDevice* device,
                Graph* graph, FunctionLibraryRuntime* flib,
                ScopedStepContainer* step_container,
                const FunctionCompiler& compiler)
      : xla_context_(xla_context),
        device_(device),
        graph_(graph),
        flib_(flib),
        step_container_(step_container),
        compiler_(compiler) {}

  // Compiles the graph. The results are written in `xla_context` that is passed
  // into the compiler.
  Status Compile();

 private:
  // NodeBinding is a wrapper on a `Node` that also contains computed
  // TensorValue.
  struct NodeBinding {
    const Node* node;
    // Kernel for this node, to be filled by CreateKernel.
    OpKernel* op_kernel;
    // Output values of this node.
    std::vector<TensorValue> tensor_values;
    // Attributes of the outputs.
    gtl::InlinedVector<AllocatorAttributes, 4> output_attrs;
  };

  // Partially sets params. This partially set params can be reused
  // across multple nodes visit.
  void PartiallySetupParams(OpKernelContext::Params* params);

  // Tests if a node is a functional node. A functional node represents a
  // defined computation and should be compiled using `compiler_`.
  bool IsFunctional(Node*);

  // Compiles a functional node and writes result to OpkernelContext. A
  // functional node represents a defined computation and should be compiled
  // using `compiler_`.
  Status CompileFunctionalNode(Node*, OpKernelContext*);

  XlaContext* xla_context_;
  XlaCompilationDevice* device_;
  Graph* graph_;
  FunctionLibraryRuntime* flib_;
  ScopedStepContainer* step_container_;
  FunctionCompiler compiler_;
  // A buffer to hold tensor inputs to a node, this is reused across the graph
  // traversal.
  gtl::InlinedVector<TensorValue, 4> tensor_inputs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_GRAPH_COMPILER_H_
