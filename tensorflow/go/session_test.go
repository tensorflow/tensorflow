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
		t1, err := NewTensor(test.input)
		if err != nil {
			t.Fatalf("NewTensor(%v): %v", test.input, err)
		}
		graph, inp, out := createTestGraph(t, t1.DataType())
		s, err := NewSession(graph, &SessionOptions{})
		if err != nil {
			t.Fatalf("NewSession() for %v: %v", test.input, err)
		}
		output, err := s.Run(map[Output]*Tensor{inp: t1}, []Output{out}, []*Operation{out.Op})
		if err != nil {
			t.Fatalf("Run() for %v: %v", test.input, err)
		}
		if len(output) != 1 {
			t.Errorf("%v: got %d outputs, want 1", test.input, len(output))
			continue
		}
		val := output[0].Value()
		if !reflect.DeepEqual(test.expected, val) {
			t.Errorf("got %v, want %v", val, test.expected)
		}
		if err := s.Close(); err != nil {
			t.Errorf("Close(): %v", err)
		}
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
