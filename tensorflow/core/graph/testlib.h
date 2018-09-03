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

// DEPRECATED: Use the C++ API defined in tensorflow/cc instead.

#ifndef TENSORFLOW_CORE_GRAPH_TESTLIB_H_
#define TENSORFLOW_CORE_GRAPH_TESTLIB_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace test {
namespace graph {

// Converts "g" into its corresponding GraphDef "def".
// DEPRECATED: call g->ToGraphDef(def) instead.
void ToGraphDef(Graph* g, GraphDef* def);

// A few helpers to construct a graph.

// Adds a node in "g" producing a constant "tensor".
Node* Constant(Graph* g, const Tensor& tensor);
Node* Constant(Graph* g, const Tensor& tensor, const string& name);

// Adds a node in "g" producing a constant "tensor" on the host.
// The given node which, unlike the regular Constant above, always
// stores its output on the host.  This is necessary for use
// in GPU tests where the test Op in question runs on the device
// but requires some arguments to be pinned to the host.
Node* HostConstant(Graph* g, const Tensor& tensor);
Node* HostConstant(Graph* g, const Tensor& tensor, const string& name);

// Adds a variable in "g" of the given "shape" and "dtype".
Node* Var(Graph* g, const DataType dtype, const TensorShape& shape);
Node* Var(Graph* g, const DataType dtype, const TensorShape& shape,
          const string& name);

// Adds an assign node in "g" which assigns "val" into "var".
Node* Assign(Graph* g, Node* var, Node* val);

// Adds a send node "g" sending "input" as a named "tensor" from
// "sender" to "receiver".
Node* Send(Graph* g, Node* input, const string& tensor, const string& sender,
           const uint64 sender_incarnation, const string& receiver);

// Adds a recv node in "g" receiving a named "tensor" from "sender"
// to "receiver".
Node* Recv(Graph* g, const string& tensor, const string& type,
           const string& sender, const uint64 sender_incarnation,
           const string& receiver);

// Adds a reduction "node" in "g" doing sum(data, axes).  "reduce" is
// a reduction, e.g., Sum, Max, Min, Mean, etc.
Node* Reduce(Graph* g, const string& reduce, Node* data, Node* axes,
             bool keep_dims = false);

// Adds a Matmul node in g doing in0.contract(in1).
Node* Matmul(Graph* g, Node* in0, Node* in1, bool transpose_a,
             bool transpose_b);

// Adds a Matmul node in g doing in0.contract(in1).
Node* BatchMatmul(Graph* g, Node* in0, Node* in1, bool adj_x, bool adj_y);

// Adds a Quantize node into g that quantize floats into QUINT8. The range of
// the input float tensor is assumed to be [-1, 1].
Node* QuantizeToUINT8(Graph* g, Node* data);

// Adds a unary function "func" "node" in "g" taking "input".
Node* Unary(Graph* g, const string& func, Node* input, int index = 0);

// Adds an identity node in "g" taking "input" and producing an
// identity copy.
Node* Identity(Graph* g, Node* input, int index = 0);

// Adds a binary function "func" node in "g" taking "in0" and "in1".
Node* Binary(Graph* g, const string& func, Node* in0, Node* in1);

// Adds a function "func" node in "g" taking inputs "ins".
Node* Multi(Graph* g, const string& func, gtl::ArraySlice<Node*> ins);

// Adds a binary add node in "g" doing in0 + in1.
Node* Add(Graph* g, Node* in0, Node* in1);

// Reverses <axis> dimensions of <tensor>>
Node* Reverse(Graph* g, Node* tensor, Node* axis);

// Generates random unit uniform distribution of the input shape.
Node* RandomUniform(Graph* g, Node* input, DataType dtype);

// Generates random unit normal distribution of the input shape.
Node* RandomGaussian(Graph* g, Node* input, DataType dtype);

// Generates random gamma distribution with the given shape and alpha[s].
// Output dtype determined by alpha.
Node* RandomGamma(Graph* g, Node* shape, Node* alpha);

// Generates random poisson distribution with the given shape and lam[s].
// Output dtype determined by lam.
Node* RandomPoisson(Graph* g, Node* shape, Node* lam);

// Rolls tensor by an offset of <shift> along the corresponding
// <axis> dimensions.
Node* Roll(Graph* g, Node* input, Node* shift, Node* axis);

// Generates random parameters from the truncated standard normal distribution
// of the nput shape
Node* TruncatedNormal(Graph* g, Node* input, DataType dtype);

// Adds an error node in "g". The node's computation always
// generates an error with the given error message "errmsg".
Node* Error(Graph* g, Node* input, const string& errmsg);

// Adds a node that generates a invalid ref output.
Node* InvalidRefType(Graph* g, DataType out_type, DataType invalid_type);

// Adds a node in "g". Its Compute() sleeps a while and outputs the
// input (i.e., same as identity).
Node* Delay(Graph* g, Node* input, Microseconds delay_micros);

// Adds a no-op "node" in "g", with control inputs from all nodes in
// control_inputs vector.
Node* NoOp(Graph* g, const std::vector<Node*>& control_inputs);

// Adds a Switch node in "g". If "in1" is true, it forwards "in0" to
// output 1. Otherwise, it forwards "in0" to output 0.
Node* Switch(Graph* g, Node* in0, Node* in1);

// Adds an Enter node in "g", which enters a new frame.
Node* Enter(Graph* g, Node* input, const string& frame_name);

// Adds an Exit node in "g", which exits a frame.
Node* Exit(Graph* g, Node* input);

// Adds a Merge node in "g" with two inputs "in0" and "in1".
Node* Merge(Graph* g, Node* in0, Node* in1);

// Adds a Merge node in "g". The first input is "in0", the remaining
// inputs are only given by their names in remaining_in.
Node* Merge(Graph* g, Node* in0, gtl::ArraySlice<string> remaining_in);

// Adds a NextIteration node in "g", which makes its input available
// to the next iteration.
Node* Next(Graph* g, const string& name, Node* input);

// Adds a LoopCond node in "g", representing the "pivot" termination
// condition of a loop.
Node* LoopCond(Graph* g, Node* input);

// Adds a less node in "g", which returns true iff "in0" < "in1".
Node* Less(Graph* g, Node* in0, Node* in1);

// Adds a select node in "g", which outputs either "inx" or "iny"
// depending on the boolean value of "c".
Node* Select(Graph* g, Node* c, Node* inx, Node* iny);

// Casts "in" into data type "dst".
Node* Cast(Graph* g, Node* in, DataType dst);

// Perform gather op on params "in0" with indices "in1" and axis "axis".
Node* Gather(Graph* g, Node* in0, Node* in1, Node* axis);

// Gets a tensor stored in the session state.
Node* GetSessionTensor(Graph* g, Node* in);

// Adds a Concat node in "g". The first input is "concat_dim", the
// dimension to concatenate on, and the tensors to concatenate are
// given in "tensors".
Node* Concat(Graph* g, Node* concat_dim, gtl::ArraySlice<Node*> tensors);

// Adds a ConcatV2 node in "g". The last input is "concat_dim", the
// dimension to concatenate on, and the tensors to concatenate are
// given in "tensors".
Node* ConcatV2(Graph* g, gtl::ArraySlice<Node*> tensors, Node* concat_dim);

// Add a Relu node in "g".
Node* Relu(Graph* g, Node* in);

// Add a Relu6 node in "g".
Node* Relu6(Graph* g, Node* in);

// Add a BiasAdd node in "g".
Node* BiasAdd(Graph* g, Node* value, Node* bias);

// Add a Conv2D node in "g".
Node* Conv2D(Graph* g, Node* in0, Node* in1);

// Add a Diag node in "g".
Node* Diag(Graph* g, Node* in, DataType type);

// Add a DiagPart node in "g".
Node* DiagPart(Graph* g, Node* in, DataType type);

}  // end namespace graph
}  // end namespace test
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_TESTLIB_H_
