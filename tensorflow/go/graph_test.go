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

import (
	"bytes"
	"fmt"
	"testing"
)

func hasOperations(g *Graph, ops ...string) error {
	var missing []string
	for _, op := range ops {
		if g.Operation(op) == nil {
			missing = append(missing, op)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf("Graph does not have the operations %v", missing)
	}

	inList := map[string]bool{}
	for _, op := range g.Operations() {
		inList[op.Name()] = true
	}

	for _, op := range ops {
		if !inList[op] {
			missing = append(missing, op)
		}
	}

	if len(missing) != 0 {
		return fmt.Errorf("Operations %v are missing from graph.Operations()", missing)
	}

	return nil
}

func TestGraphWriteToAndImport(t *testing.T) {
	// Construct a graph
	g := NewGraph()
	v, err := NewTensor(int64(1))
	if err != nil {
		t.Fatal(err)
	}
	input, err := Placeholder(g, "input", v.DataType())
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Neg(g, "neg", input); err != nil {
		t.Fatal(err)
	}

	// Serialize the graph
	buf := new(bytes.Buffer)
	if _, err := g.WriteTo(buf); err != nil {
		t.Fatal(err)
	}

	// Import it into the same graph, with a prefix
	if err := g.Import(buf.Bytes(), "imported"); err != nil {
		t.Error(err)
	}
	if err := hasOperations(g, "input", "neg", "imported/input", "imported/neg"); err != nil {
		t.Error(err)
	}
}
