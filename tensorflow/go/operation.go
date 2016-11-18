// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"errors"
	"unsafe"
)

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

// Shape returns the (possibly incomplete) shape of the tensor produced p.
//
// Returns a slice of length 0 if the tensor is a scalar.  Returns a slice
// where shape[i] is the size of the i-th dimension of the tensor, or -1 if the
// size of that dimension is not known.
//
// Returns an error if the number of dimensions of the tensor is not known.
func (p Output) Shape() (shape []int64, err error) {
	status := newStatus()
	port := p.c()
	ndims := C.TF_GraphGetTensorNumDims(p.Op.g.c, port, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	if ndims < 0 {
		return nil, errors.New("unknown number of dimensions")
	}
	if ndims == 0 {
		return nil, nil
	}
	dims := make([]C.int64_t, ndims)
	C.TF_GraphGetTensorShape(p.Op.g.c, port, &dims[0], ndims, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	ret := make([]int64, ndims)
	for i := 0; i < int(ndims); i++ {
		ret[i] = int64(dims[i])
	}
	return ret, nil
}

func (p Output) c() C.TF_Output {
	return C.TF_Output{oper: p.Op.c, index: C.int(p.Index)}
}

func (p Output) canBeAnInput() {}

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
