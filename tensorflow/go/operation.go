/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import "unsafe"

// Operation that has been added to the graph.
type Operation struct {
	c *C.TF_Operation
	// A reference to the Graph to prevent it from
	// being GCed while the Operation is still alive.
	g *Graph
}

// Name returns the name of the operation.
func (op *Operation) Name() string {
	return C.GoString(C.TF_OperationName(op.c))
}

// Type returns the name of the operator used by this operation.
func (op *Operation) Type() string {
	return C.GoString(C.TF_OperationOpType(op.c))
}

// NumOutputs returns the number of outputs of op.
func (op *Operation) NumOutputs() int {
	return int(C.TF_OperationNumOutputs(op.c))
}

// Device returns a specification of the device on which this operation
// will be executed, or the empty string if there is no such specification.
func (op *Operation) Device() string {
	return C.GoString(C.TF_OperationDevice(op.c))
}

// OutputListSize returns the size of the list of Outputs that is produced by a
// named output of op.
//
// An Operation has multiple named outputs, each of which produces either
// a single tensor or a list of tensors. This method returns the size of
// the list of tensors for a specific output of the operation, identified
// by its name.
func (op *Operation) OutputListSize(output string) (int, error) {
	cname := C.CString(output)
	defer C.free(unsafe.Pointer(cname))
	status := newStatus()
	n := C.TF_OperationOutputListLength(op.c, cname, status.c)
	return int(n), status.Err()
}

// Output returns the i-th output of op.
func (op *Operation) Output(i int) Output {
	return Output{op, i}
}

// NumInputs returns the number of inputs of op.
func (op *Operation) NumInputs() int {
	return int(C.TF_OperationNumInputs(op.c))
}

// Output represents one of the outputs of an operation in the graph. Has a
// DataType (and eventually a Shape).  May be passed as an input argument to a
// function for adding operations to a graph, or to a Session's Run() method to
// fetch that output as a tensor.
type Output struct {
	// Op is the Operation that produces this Output.
	Op *Operation

	// Index specifies the index of the output within the Operation.
	Index int
}

// DataType returns the type of elements in the tensor produced by p.
func (p Output) DataType() DataType {
	return DataType(C.TF_OperationOutputType(p.c()))
}

// Shape returns the (possibly incomplete) shape of the tensor produced p.
func (p Output) Shape() Shape {
	status := newStatus()
	port := p.c()
	ndims := C.TF_GraphGetTensorNumDims(p.Op.g.c, port, status.c)
	if err := status.Err(); err != nil {
		// This should not be possible since an error only occurs if
		// the operation does not belong to the graph.  It should not
		// be possible to construct such an Operation object.
		return Shape{}
	}
	if ndims < 0 {
		return Shape{}
	}
	if ndims == 0 {
		return ScalarShape()
	}
	dims := make([]C.int64_t, ndims)
	C.TF_GraphGetTensorShape(p.Op.g.c, port, &dims[0], ndims, status.c)
	if err := status.Err(); err != nil {
		// Same as above, should not be possible.
		return Shape{}
	}
	ret := Shape{dims: make([]int64, ndims)}
	for i := 0; i < int(ndims); i++ {
		ret.dims[i] = int64(dims[i])
	}
	return ret
}

func (p Output) c() C.TF_Output {
	if p.Op == nil {
		// Attempt to provide a more useful panic message than "nil
		// pointer dereference".
		panic("nil-Operation. If the Output was created with a Scope object, see Scope.Err() for details.")
	}
	return C.TF_Output{oper: p.Op.c, index: C.int(p.Index)}
}

func (p Output) canBeAnInput() {}

// Consumers returns the inputs that consume this output.
func (p Output) Consumers() []Consumer {
	max := int(C.TF_OperationOutputNumConsumers(p.c()))
	if max == 0 {
		return nil
	}
	inputs := make([]C.TF_Input, max)
	n := C.TF_OperationOutputConsumers(p.c(), (*C.TF_Input)(unsafe.Pointer(&inputs[0])), C.int(max))
	inputs = inputs[:int(n)]

	var consumers []Consumer
	for _, consumer := range inputs {
		consumers = append(consumers, Consumer{
			Index: int(consumer.index),
			Op: &Operation{
				c: consumer.oper,
				g: p.Op.g,
			},
		})
	}

	return consumers
}

// Consumer identifies a specific input of an operation that consumes the output
// of another operation.
type Consumer struct {
	// Op is the Operation that is consuming the output of another operation.
	Op *Operation

	// Index is the index of the input within Op that the output of another
	// operation is connected to.
	Index int
}

func (p Consumer) c() C.TF_Input {
	if p.Op == nil {
		// Attempt to provide a more useful panic message than "nil
		// pointer dereference".
		panic("nil-Operation. Consumer objects should only be created by a call to Output.Consumers")
	}
	return C.TF_Input{oper: p.Op.c, index: C.int(p.Index)}
}

// DataType returns the type of the input.
func (p Consumer) DataType() DataType {
	return DataType(C.TF_OperationInputType(p.c()))
}

// Producer returns the Output that is connected to this Consumer.
func (p Consumer) Producer() Output {
	output := C.TF_OperationInput(p.c())
	return Output{
		Op: &Operation{
			c: output.oper,
			g: p.Op.g,
		},
		Index: int(output.index),
	}
}

// Input is the interface for specifying inputs to an operation being added to
// a Graph.
//
// Operations can have multiple inputs, each of which could be either a tensor
// produced by another operation (an Output object), or a list of tensors
// produced by other operations (an OutputList). Thus, this interface is
// implemented by both Output and OutputList.
//
// See OpSpec.Input for more information.
type Input interface {
	// Unexported to preclude implementations outside this package.
	canBeAnInput()
}

// OutputList represents a list of Outputs that can be provided as input to
// another operation.
type OutputList []Output

func (l OutputList) canBeAnInput() {}
