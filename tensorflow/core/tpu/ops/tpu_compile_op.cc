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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

REGISTER_OP("_TPUCompileMlir")
    .Attr("num_computations: int >= 0")
    .Attr("mlir_module: string=\"\"")
    .Attr("metadata: string")
    .Attr("NumDynamicShapes: int >= 0")
    // Do not try to optimize me away. We would like the compilation-op to be
    // invoked for every step, and not be constant-folded away, in case the
    // program is evicted from the compilation cache.
    .SetIsStateful()
    .Input("dynamic_shapes: NumDynamicShapes * int64")
    .Output("compilation_status: string")
    .Output("program: num_computations * string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_computations;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(c->attrs(), "num_computations", &num_computations));
      // Compilation status.
      c->set_output(0, c->Scalar());
      // Programs.
      for (int i = 0; i < num_computations; ++i) {
        c->set_output(i + 1, c->Vector(3));
      }
      return Status::OK();
    })
    .Doc(
        R"(
Compiles a computations for execution on one or more TPU devices.
For the internal use of the distributed TPU compiler.

'mlir_module' is a serialized MLIR module with a `main` function that contains
target computation.
'dynamic_shapes' contains dynamic shapes of arguments whose shapes were not
known statically at TPUReplication rewrite time.
'metadata' is a serialized TPUCompileMetadataProto describing the shapes and
types of the inputs to the computation, as well as a mapping onto the TPU pod
topology.
'program' output is a string key that is passed to the TPUExecute op and used to
look up the program in the compilation cache.
)");

REGISTER_OP("_TPUCompileMlirPlaceholderProgramKey")
    .SetIsStateful()
    .Output("program: string")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(3));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(
        R"(
Placeholder program key (compilation cache key) of a _TPUCompileMlir `program`.

This op can be used when certain rewrite passes materialize ops that require a
program key but the _TPUCompileMlir op has not been added yet. Subsequent
rewrite passes must replace this op with a _TPUCompileMlir op `program` output.
)");

REGISTER_OP("TPUCompile")
    .Attr("num_computations: int >= 0")
    .Attr("function: func")
    .Attr("metadata: string")
    .Attr("NumDynamicShapes: int >= 0")
    .Attr("Tguaranteed_constants: list(type) >= 0")
    // Do not try to optimize me away. We would like the compilation-op to be
    // invoked for every step, and not be constant-folded away, in case the
    // program is evicted from the compilation cache.
    .SetIsStateful()
    .Input("dynamic_shapes: NumDynamicShapes * int64")
    .Input("guaranteed_constants: Tguaranteed_constants")
    .Output("compilation_status: string")
    .Output("program: num_computations * string")
    .Output("may_modify_variables: num_computations * bool")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      int num_computations;
      TF_RETURN_IF_ERROR(
          GetNodeAttr(c->attrs(), "num_computations", &num_computations));
      // Compilation status.
      c->set_output(0, c->Scalar());
      // Programs.
      for (int i = 0; i < num_computations; ++i) {
        c->set_output(i + 1, c->Vector(3));
      }
      // May modify variables.
      for (int i = 0; i < num_computations; ++i) {
        c->set_output(num_computations + i + 1, c->Scalar());
      }
      return Status::OK();
    });

REGISTER_OP("TPUCompileSucceededAssert")
    .Input("compilation_status: string")
    // Do not optimize me away. Read the comment on TPUCompileOp for more
    // details.
    .SetIsStateful()
    .SetShapeFn(shape_inference::NoOutputs);

}  // namespace tensorflow
