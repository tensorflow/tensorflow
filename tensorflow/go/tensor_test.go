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

func TestNewTensor(t *testing.T) {
	var tests = []struct {
		shape []int64
		value interface{}
	}{
		{nil, int8(5)},
		{nil, int16(5)},
		{nil, int32(5)},
		{nil, int64(5)},
		{nil, int64(5)},
		{nil, uint8(5)},
		{nil, uint16(5)},
		{nil, float32(5)},
		{nil, float64(5)},
		{nil, complex(float32(5), float32(6))},
		{nil, complex(float64(5), float64(6))},
		{[]int64{1}, []float64{1}},
		{[]int64{1}, [1]float64{1}},
		{[]int64{3, 2}, [][]float64{{1, 2}, {3, 4}, {5, 6}}},
		{[]int64{2, 3}, [2][3]float64{{1, 2, 3}, {3, 4, 6}}},
		{[]int64{4, 3, 2}, [][][]float64{
			{{1, 2}, {3, 4}, {5, 6}},
			{{7, 8}, {9, 10}, {11, 12}},
			{{0, -1}, {-2, -3}, {-4, -5}},
			{{-6, -7}, {-8, -9}, {-10, -11}},
		}},
		{[]int64{2, 0}, [][]int64{{}, {}}},
	}

	var errorTests = []interface{}{
		struct{ a int }{5},
		new(int32),
		new([]int32),
		// native ints not supported
		int(5),
		[]int{5},
		// uint32 and uint64 are not supported in TensorFlow
		uint32(5),
		[]uint32{5},
		uint64(5),
		[]uint64{5},
		// Mismatched dimensions
		[][]float32{{1, 2, 3}, {4}},
	}

	for _, test := range tests {
		tensor, err := NewTensor(test.value)
		if err != nil {
			t.Errorf("NewTensor(%v): %v", test.value, err)
			continue
		}
		if !reflect.DeepEqual(test.shape, tensor.Shape()) {
			t.Errorf("Tensor.Shape(): got %v, want %v", tensor.Shape(), test.shape)
		}

		// Test that encode and decode gives the same value. We skip arrays because
		// they're returned as slices.
		if reflect.TypeOf(test.value).Kind() != reflect.Array {
			got := tensor.Value()
			if !reflect.DeepEqual(test.value, got) {
				t.Errorf("encode/decode: got %v, want %v", got, test.value)
			}
		}
	}

	for _, test := range errorTests {
		tensor, err := NewTensor(test)
		if err == nil {
			t.Errorf("NewTensor(%v): %v", test, err)
		}
		if tensor != nil {
			t.Errorf("NewTensor(%v) = %v, want nil", test, tensor)
		}
	}
}

func benchmarkNewTensor(b *testing.B, v interface{}) {
	for i := 0; i < b.N; i++ {
		if t, err := NewTensor(v); err != nil || t == nil {
			b.Fatalf("(%v, %v)", t, err)
		}
	}
}

func BenchmarkNewTensor(b *testing.B) {
	var (
		// Some sample sizes from the Inception image labeling model.
		// Where input tensors correspond to a 224x224 RGB image
		// flattened into a vector.
		vector [224 * 224 * 3]int32
	)
	b.Run("[150528]", func(b *testing.B) { benchmarkNewTensor(b, vector) })
}
