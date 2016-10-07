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

// #include "tensorflow/c/c_api.h"
//
// #include <stdlib.h>
// #include <string.h>
import "C"

import (
	"fmt"
	"io"
	"runtime"
	"unsafe"
)

// Graph represents a computation graph. Graphs may be shared between sessions.
type Graph struct {
	c *C.TF_Graph
}

// NewGraph returns a new Graph.
func NewGraph() *Graph {
	g := &Graph{C.TF_NewGraph()}
	runtime.SetFinalizer(g, (*Graph).finalizer)
	return g
}

func (g *Graph) finalizer() {
	C.TF_DeleteGraph(g.c)
}

// WriteTo writes out a serialized representation of g to w.
//
// Implements the io.WriterTo interface.
func (g *Graph) WriteTo(w io.Writer) (int64, error) {
	buf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(buf)
	status := newStatus()
	C.TF_GraphToGraphDef(g.c, buf, status.c)
	if err := status.Err(); err != nil {
		return 0, err
	}
	if buf.length > (1 << 30) {
		// For very large graphs, the writes can be chunked.
		// Punt on that for now.
		return 0, fmt.Errorf("Graph is too large to write out, Graph.WriteTo needs to be updated")
	}
	// A []byte slice backed by C memory.
	// See: https://github.com/golang/go/wiki/cgo#turning-c-arrays-into-go-slices
	length := int(buf.length)
	slice := (*[1 << 30]byte)(unsafe.Pointer(buf.data))[:length:length]
	n, err := w.Write(slice)
	return int64(n), err
}

// Import imports the nodes and edges from a serialized representation of
// another Graph into g.
//
// Names of imported nodes will be prefixed with prefix.
func (g *Graph) Import(def []byte, prefix string) error {
	cprefix := C.CString(prefix)
	defer C.free(unsafe.Pointer(cprefix))

	opts := C.TF_NewImportGraphDefOptions()
	defer C.TF_DeleteImportGraphDefOptions(opts)
	C.TF_ImportGraphDefOptionsSetPrefix(opts, cprefix)

	buf := C.TF_NewBuffer()
	defer C.TF_DeleteBuffer(buf)
	// Would have preferred to use C.CBytes, but that does not play well
	// with "go vet" till https://github.com/golang/go/issues/17201 is
	// resolved.
	buf.length = C.size_t(len(def))
	buf.data = C.malloc(buf.length)
	if buf.data == nil {
		return fmt.Errorf("unable to allocate memory")
	}
	defer C.free(buf.data)
	C.memcpy(buf.data, unsafe.Pointer(&def[0]), buf.length)

	status := newStatus()
	C.TF_GraphImportGraphDef(g.c, buf, opts, status.c)
	if err := status.Err(); err != nil {
		return err
	}
	return nil
}

// Operation returns the Operation named name in the Graph, or nil if no such
// operation is present.
func (g *Graph) Operation(name string) *Operation {
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))
	cop := C.TF_GraphOperationByName(g.c, cname)
	if cop == nil {
		return nil
	}
	return &Operation{cop,g}
}
