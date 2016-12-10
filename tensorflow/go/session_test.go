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

import (
	"fmt"
	"reflect"
	"testing"
)

func createTestGraph(t *testing.T, dt DataType) (*Graph, Output, Output) {
	g := NewGraph()
	inp, err := Placeholder(g, "p1", dt)
	if err != nil {
		t.Fatalf("Placeholder() for %v: %v", dt, err)
	}
	out, err := Neg(g, "neg1", inp)
	if err != nil {
		t.Fatalf("Neg() for %v: %v", dt, err)
	}
	return g, inp, out
}

func TestSessionRunNeg(t *testing.T) {
	var tests = []struct {
		input    interface{}
		expected interface{}
	}{
		{int64(1), int64(-1)},
		{[]float64{-1, -2, 3}, []float64{1, 2, -3}},
		{[][]float32{{1, -2}, {-3, 4}}, [][]float32{{-1, 2}, {3, -4}}},
	}

	for _, test := range tests {
		t.Run(fmt.Sprint(test.input), func(t *testing.T) {
			t1, err := NewTensor(test.input)
			if err != nil {
				t.Fatal(err)
			}
			graph, inp, out := createTestGraph(t, t1.DataType())
			s, err := NewSession(graph, &SessionOptions{})
			if err != nil {
				t.Fatal(err)
			}
			output, err := s.Run(map[Output]*Tensor{inp: t1}, []Output{out}, []*Operation{out.Op})
			if err != nil {
				t.Fatal(err)
			}
			if len(output) != 1 {
				t.Fatalf("got %d outputs, want 1", len(output))
			}
			val := output[0].Value()
			if !reflect.DeepEqual(test.expected, val) {
				t.Errorf("got %v, want %v", val, test.expected)
			}
			if err := s.Close(); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestSessionRunConcat(t *testing.T) {
	// Runs the Concat operation on two matrices: m1 and m2, along the
	// first dimension (dim1).
	// This tests the use of both Output and OutputList as inputs to the
	// Concat operation.
	var (
		g       = NewGraph()
		dim1, _ = Const(g, "dim1", int32(1))
		m1, _   = Const(g, "m1", [][]int64{
			{1, 2, 3},
			{4, 5, 6},
		})
		m2, _ = Const(g, "m2", [][]int64{
			{7, 8, 9},
			{10, 11, 12},
		})
		want = [][]int64{
			{1, 2, 3, 7, 8, 9},
			{4, 5, 6, 10, 11, 12},
		}
	)
	concat, err := g.AddOperation(OpSpec{
		Type: "Concat",
		Input: []Input{
			dim1,
			OutputList{m1, m2},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewSession(g, &SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []Output{concat.Output(0)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 1 {
		t.Fatal(len(output))
	}
	if got := output[0].Value(); !reflect.DeepEqual(got, want) {
		t.Fatalf("Got %v, want %v", got, want)
	}
}

func TestSessionWithStringTensors(t *testing.T) {
	// Construct the graph:
	// AsString(StringToHashBucketFast("PleaseHashMe")) Will be much
	// prettier if using the ops package, but in this package graphs are
	// constructed from first principles.
	var (
		g       = NewGraph()
		feed, _ = Const(g, "input", "PleaseHashMe")
		hash, _ = g.AddOperation(OpSpec{
			Type:  "StringToHashBucketFast",
			Input: []Input{feed},
			Attrs: map[string]interface{}{
				"num_buckets": int64(1 << 32),
			},
		})
		str, _ = g.AddOperation(OpSpec{
			Type:  "AsString",
			Input: []Input{hash.Output(0)},
		})
	)
	s, err := NewSession(g, nil)
	if err != nil {
		t.Fatal(err)
	}
	output, err := s.Run(nil, []Output{str.Output(0)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(output) != 1 {
		t.Fatal(len(output))
	}
	got, ok := output[0].Value().(string)
	if !ok {
		t.Fatalf("Got %T, wanted string", output[0].Value())
	}
	if want := "1027741475"; got != want {
		t.Fatalf("Got %q, want %q", got, want)
	}
}

func TestConcurrency(t *testing.T) {
	tensor, err := NewTensor(int64(1))
	if err != nil {
		t.Fatalf("NewTensor(): %v", err)
	}

	graph, inp, out := createTestGraph(t, tensor.DataType())
	s, err := NewSession(graph, &SessionOptions{})
	if err != nil {
		t.Fatalf("NewSession(): %v", err)
	}
	for i := 0; i < 100; i++ {
		// Session may close before Run() starts, so we don't check the error.
		go s.Run(map[Output]*Tensor{inp: tensor}, []Output{out}, []*Operation{out.Op})
	}
	if err = s.Close(); err != nil {
		t.Errorf("Close() 1: %v", err)
	}
	if err = s.Close(); err != nil {
		t.Errorf("Close() 2: %v", err)
	}
}
