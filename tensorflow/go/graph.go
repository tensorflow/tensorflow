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
import "C"

import (
	"runtime"
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
