/*
Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
	"math"
	"testing"
)

func TestSavedModelHalfPlusTwo(t *testing.T) {
	var (
		exportDir = "testdata/saved_model/half_plus_two/00000123"
		tags      = []string{"serve"}
		options   = new(SessionOptions)
	)

	// Load saved model half_plus_two.
	m, err := LoadSavedModel(exportDir, tags, options)
	if err != nil {
		t.Fatalf("LoadSavedModel(): %v", err)
	}

	// Check that named operations x and y are present in the graph.
	if op := m.Graph.Operation("x"); op == nil {
		t.Fatalf("\"x\" not found in graph")
	}
	if op := m.Graph.Operation("y"); op == nil {
		t.Fatalf("\"y\" not found in graph")
	}

	// Define test cases for half plus two (y = 0.5 * x + 2).
	tests := []struct {
		name string
		X    float32
		Y    float32
	}{
		{"NegVal", -1, 1.5},
		{"PosVal", 1, 2.5},
		{"Zero", 0, 2.0},
		{"NegInf", float32(math.Inf(-1)), float32(math.Inf(-1))},
		{"PosInf", float32(math.Inf(1)), float32(math.Inf(1))},
	}

	// Run tests.
	for _, c := range tests {
		t.Run(c.name, func(t *testing.T) {
			x, err := NewTensor([]float32{c.X})
			if err != nil {
				t.Fatal(err)
			}

			y, err := m.Session.Run(
				map[Output]*Tensor{
					m.Graph.Operation("x").Output(0): x,
				},
				[]Output{
					m.Graph.Operation("y").Output(0),
				},
				nil,
			)
			if err != nil {
				t.Fatal(err)
			}

			got := y[0].Value().([]float32)[0]
			if got != c.Y {
				t.Fatalf("got: %#v, want: %#v", got, c.Y)
			}
		})
	}

	t.Logf("SavedModel: %+v", m)
	// TODO(jhseu): half_plus_two has a tf.Example proto dependency to run.
	// Add a more thorough test when the generated protobufs are available.
}

func TestSavedModelWithEmptyTags(t *testing.T) {
	var (
		exportDir = "testdata/saved_model/half_plus_two_empty_tags/00000123"
		tags      = []string{}
		options   = new(SessionOptions)
	)

	m, err := LoadSavedModel(exportDir, tags, options)
	if err != nil {
		t.Fatalf("LoadSavedModel() failed with an empty tags set: %v", err)
	}

	if op := m.Graph.Operation("x"); op == nil {
		t.Fatalf("\"x\" not found in graph")
	}

	t.Logf("Model loaded successfully with an empty tags set: %+v", m)
}
