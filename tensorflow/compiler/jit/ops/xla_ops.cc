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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::InferenceContext;

REGISTER_OP("XlaLaunch")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Attr("function: func")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of XlaLaunch.
    .SetIsStateful()
    .Doc("XLA Launch Op. For use by the XLA JIT only.");

REGISTER_OP("XlaLaunchV2")
    .Input("args: Targs")
    .Output("results: Tresults")
    .Attr("Targs: list(type) >= 0")
    .Attr("Tresults: list(type) >= 0")
    .Attr("constants: list(int) >= 0")
    .Attr("resources: list(int) >= 0")
    .Attr("function: func")
    .Doc("XLA Launch Op. For use by the XLA JIT only.");

REGISTER_OP("XlaClusterOutput")
    .Input("input: T")
    // Note: when replication is supported, this op will have N outputs.
    .Output("outputs: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      for (int i = 0; i < c->num_outputs(); ++i) {
        c->set_output(i, c->input(0));
      }
      return OkStatus();
    })
    .Doc(
        "Operator that connects the output of an XLA computation to other "
        "consumer graph nodes.");

REGISTER_OP("_XlaCompile")
    .Input("constants: Tconstants")
    .Attr("Tconstants: list(type) >= 0")
    .Attr("must_compile: bool")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Input("resources: Nresources * resource")
    .Attr("Nresources: int >= 0")
    .Output("key: string")
    .Output("compilation_successful: bool")
    .Attr("function: func")
    // The compilation cache is stateful.
    .SetIsStateful()
    .Doc(R"(XLA Compile Op. For use by the XLA JIT only.

Compiles a TensorFlow function into an XLA LocalExecutable and returns a key
that _XlaRun can use to look up the LocalExecutable and execute it.

key: A key that can be used to look up the local executable compiled by the
   node and associated metadata.

compilation_successful: If the `must_compile` attr is false the _XlaCompile op
   can decide not to compile the clusters based on some profitability
   heuristics.  In that case `compilation_successful` is false if _XlaCompile
   chose not to compile the cluster.  If the `must_compile` attr is true then
   _XlaCompile always attempts to compile the cluster and
   `compilation_successful` is always true.
)");

REGISTER_OP("_XlaRun")
    .Input("args: Targs")
    .Attr("Targs: list(type) >= 0")
    .Output("results: Tresults")
    .Attr("Tresults: list(type) >= 0")
    .Input("key: string")
    // XLA random-number generation ops are stateful.
    // TODO(phawkins): create stateful and non-stateful variants of _XlaRun.
    .SetIsStateful()
    .Doc(R"(XLA Run Op. For use by the XLA JIT only.

Executes a TensorFlow function previously compiled into a LocalExecutable by an
_XlaCompile op.
)");

REGISTER_OP("_XlaMerge")
    .Input("partitioned_call: T")
    .Input("xla_run: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      return OkStatus();
    })
    .Doc(R"(XLA Merge Op. For use by the XLA JIT only.

Merges the outputs from the PartitionedCall node and the _XlaRun node.
Unlike the TensorFlow Merge op, which requires inputs of some types to be
placed on the host, the _XlaMerge op can merge inputs of all types when
placed on the device. This prevents the need for copy operations, in
particular when an XLA cluster has int32 outputs. The _XlaMerge up does not
have a value_index output that identifies the chosen input.
)");

}  // namespace tensorflow
