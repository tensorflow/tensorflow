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
	"fmt"
	"runtime"
	"runtime/debug"
	"testing"
)

// createGraphAndOp creates an Operation but loses the reference to the Graph.
func createGraphAndOp() (*Operation, error) {
	t, err := NewTensor(int64(1))
	if err != nil {
		return nil, err
	}
	g := NewGraph()
	output, err := Placeholder(g, "my_placeholder", t.DataType())
	if err != nil {
		return nil, err
	}
	return output.Op, nil
}

func TestOperationLifetime(t *testing.T) {
	// Ensure that the Graph is not garbage collected while the program
	// still has access to the Operation.
	op, err := createGraphAndOp()
	if err != nil {
		t.Fatal(err)
	}
	forceGC()
	if got, want := op.Name(), "my_placeholder"; got != want {
		t.Errorf("Got '%s', want '%s'", got, want)
	}
	if got, want := op.Type(), "Placeholder"; got != want {
		t.Errorf("Got '%s', want '%s'", got, want)
	}
}

func TestOperationOutputListSize(t *testing.T) {
	graph := NewGraph()
	c1, err := Const(graph, "c1", int64(1))
	if err != nil {
		t.Fatal(err)
	}
	c2, err := Const(graph, "c2", [][]int64{{1, 2}, {3, 4}})
	if err != nil {
		t.Fatal(err)
	}
	// The ShapeN op takes a list of tensors as input and a list as output.
	op, err := graph.AddOperation(OpSpec{
		Type:  "ShapeN",
		Input: []Input{OutputList{c1, c2}},
	})
	if err != nil {
		t.Fatal(err)
	}
	n, err := op.OutputListSize("output")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := n, 2; got != want {
		t.Errorf("Got %d, want %d", got, want)
	}
	if got, want := op.NumOutputs(), 2; got != want {
		t.Errorf("Got %d, want %d", got, want)
	}
}

func TestOperationShapeAttribute(t *testing.T) {
	g := NewGraph()
	_, err := g.AddOperation(OpSpec{
		Type: "Placeholder",
		Attrs: map[string]interface{}{
			"dtype": Float,
			"shape": MakeShape(-1, 3),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	// If and when the API to get attributes is added, check that here.
}

func TestOutputDataTypeAndShape(t *testing.T) {
	graph := NewGraph()
	testdata := []struct {
		Value interface{}
		Shape []int64
		dtype DataType
	}{
		{ // Scalar
			int64(0),
			[]int64{},
			Int64,
		},
		{ // Vector
			[]int32{1, 2, 3},
			[]int64{3},
			Int32,
		},
		{ // Matrix
			[][]float64{
				{1, 2, 3},
				{4, 5, 6},
			},
			[]int64{2, 3},
			Double,
		},
	}
	for idx, test := range testdata {
		t.Run(fmt.Sprintf("#%d Value %T", idx, test.Value), func(t *testing.T) {
			c, err := Const(graph, fmt.Sprintf("const%d", idx), test.Value)
			if err != nil {
				t.Fatal(err)
			}
			if got, want := c.DataType(), test.dtype; got != want {
				t.Errorf("Got DataType %v, want %v", got, want)
			}
			shape := c.Shape()
			if got, want := shape.NumDimensions(), len(test.Shape); got != want {
				t.Fatalf("Got a shape with %d dimensions, want %d", got, want)
			}
			for i := 0; i < len(test.Shape); i++ {
				if got, want := shape.Size(i), test.Shape[i]; got != want {
					t.Errorf("Got %d, want %d for dimension #%d/%d", got, want, i, len(test.Shape))
				}
			}
		})
	}
	// Unknown number of dimensions
	dummyTensor, err := NewTensor(float64(0))
	if err != nil {
		t.Fatal(err)
	}
	placeholder, err := Placeholder(graph, "placeholder", dummyTensor.DataType())
	if err != nil {
		t.Fatal(err)
	}
	if shape := placeholder.Shape(); shape.NumDimensions() != -1 {
		t.Errorf("Got shape %v, wanted an unknown number of dimensions", shape)
	}
}

func forceGC() {
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	// It was empirically observed that without this extra allocation
	// TestOperationLifetime would fail only 50% of the time if
	// Operation did not hold on to a reference to Graph. With this
	// additional allocation, and with the bug where Operation does
	// not hold onto a Graph, the test failed 90+% of the time.
	//
	// The author is aware that this technique is potentially fragile
	// and fishy. Suggestions for alternatives are welcome.
	bytesTillGC := mem.NextGC - mem.HeapAlloc + 1
	_ = make([]byte, bytesTillGC)
	runtime.GC()
	debug.FreeOSMemory()
}
