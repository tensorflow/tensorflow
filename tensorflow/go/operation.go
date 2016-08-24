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
import "unsafe"

// Operation that has been added to the graph.
type Operation struct {
	c *C.TF_Operation
}

// Port represents a specific input or output of an operation, e.g. to specify
// the specific output to pass as an input to a new op.
//
// Note the difference in naming convention: Port corresponds to Tensor/Output
// in the Python API.
type Port struct {
	Op    *Operation
	Index int
}

func (p *Port) c() C.TF_Port {
	return C.TF_Port{oper: p.Op.c, index: C.int(p.Index)}
}

// opBuilder is for use by the generated op code to create new Operations.
// Build() must be called for any in-progress Operation, or else we leak.
type opBuilder struct {
	c *C.TF_OperationDescription
}

func newOpBuilder(g *Graph, typ string, name string) *opBuilder {
	opType := C.CString(typ)
	opName := C.CString(name)
	b := &opBuilder{c: C.TF_NewOperation(g.c, opType, opName)}
	C.free(unsafe.Pointer(opType))
	C.free(unsafe.Pointer(opName))
	return b
}

func (b *opBuilder) SetAttrTensor(name string, t *Tensor) error {
	status := newStatus()
	attrName := C.CString(name)
	C.TF_SetAttrTensor(b.c, attrName, t.c(), status.c)
	C.free(unsafe.Pointer(attrName))
	return status.Err()
}

func (b *opBuilder) SetAttrType(name string, typ DataType) {
	attrName := C.CString(name)
	C.TF_SetAttrType(b.c, attrName, C.TF_DataType(typ))
	C.free(unsafe.Pointer(attrName))
}

func (b *opBuilder) AddInput(port Port) {
	C.TF_AddInput(b.c, port.c())
}

func (b *opBuilder) Build() (*Operation, error) {
	status := newStatus()
	op := &Operation{c: C.TF_FinishOperation(b.c, status.c)}
	if err := status.Err(); err != nil {
		return nil, err
	}
	b.c = nil
	return op, nil
}
