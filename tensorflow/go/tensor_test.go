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
	"bytes"
	"reflect"
	"testing"
)

func TestNewTensor(t *testing.T) {
	var tests = []struct {
		shape []int64
		value interface{}
	}{
		{nil, bool(true)},
		{nil, int8(5)},
		{nil, int16(5)},
		{nil, int32(5)},
		{nil, int64(5)},
		{nil, uint8(5)},
		{nil, uint16(5)},
		{nil, float32(5)},
		{nil, float64(5)},
		{nil, complex(float32(5), float32(6))},
		{nil, complex(float64(5), float64(6))},
		{nil, "a string"},
		{[]int64{2}, []bool{true, false}},
		{[]int64{1}, []float64{1}},
		{[]int64{1}, [1]float64{1}},
		{[]int64{2}, []string{"string", "slice"}},
		{[]int64{2}, [2]string{"string", "array"}},
		{[]int64{3, 2}, [][]float64{{1, 2}, {3, 4}, {5, 6}}},
		{[]int64{2, 3}, [2][3]float64{{1, 2, 3}, {3, 4, 6}}},
		{[]int64{4, 3, 2}, [][][]float64{
			{{1, 2}, {3, 4}, {5, 6}},
			{{7, 8}, {9, 10}, {11, 12}},
			{{0, -1}, {-2, -3}, {-4, -5}},
			{{-6, -7}, {-8, -9}, {-10, -11}},
		}},
		{[]int64{2, 0}, [][]int64{{}, {}}},
		{[]int64{2, 2}, [][]string{{"row0col0", "row0,col1"}, {"row1col0", "row1,col1"}}},
		{[]int64{2, 3}, [2][3]string{
			{"row0col0", "row0,col1", "row0,col2"},
			{"row1col0", "row1,col1", "row1,col2"},
		}},
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

func TestTensorSerialization(t *testing.T) {
	var tests = []interface{}{
		bool(true),
		int8(5),
		int16(5),
		int32(5),
		int64(5),
		uint8(5),
		uint16(5),
		float32(5),
		float64(5),
		complex(float32(5), float32(6)),
		complex(float64(5), float64(6)),
		[]float64{1},
		[][]float32{{1, 2}, {3, 4}, {5, 6}},
		[][][]int8{
			{{1, 2}, {3, 4}, {5, 6}},
			{{7, 8}, {9, 10}, {11, 12}},
			{{0, -1}, {-2, -3}, {-4, -5}},
			{{-6, -7}, {-8, -9}, {-10, -11}},
		},
		[]bool{true, false, true},
	}
	for _, v := range tests {
		t1, err := NewTensor(v)
		if err != nil {
			t.Errorf("(%v): %v", v, err)
			continue
		}
		buf := new(bytes.Buffer)
		n, err := t1.WriteContentsTo(buf)
		if err != nil {
			t.Errorf("(%v): %v", v, err)
			continue
		}
		if n != int64(buf.Len()) {
			t.Errorf("(%v): WriteContentsTo said it wrote %v bytes, but wrote %v", v, n, buf.Len())
		}
		t2, err := ReadTensor(t1.DataType(), t1.Shape(), buf)
		if err != nil {
			t.Errorf("(%v): %v", v, err)
			continue
		}
		if buf.Len() != 0 {
			t.Errorf("(%v): %v bytes written by WriteContentsTo not read by ReadTensor", v, buf.Len())
		}
		if got, want := t2.DataType(), t1.DataType(); got != want {
			t.Errorf("(%v): Got %v, want %v", v, got, want)
		}
		if got, want := t2.Shape(), t1.Shape(); !reflect.DeepEqual(got, want) {
			t.Errorf("(%v): Got %v, want %v", v, got, want)
		}
		if got, want := t2.Value(), v; !reflect.DeepEqual(got, want) {
			t.Errorf("(%v): Got %v, want %v", v, got, want)
		}
	}
}

func TestReadTensorDoesNotReadBeyondContent(t *testing.T) {
	t1, _ := NewTensor(int8(7))
	t2, _ := NewTensor(float32(2.718))
	buf := new(bytes.Buffer)
	if _, err := t1.WriteContentsTo(buf); err != nil {
		t.Fatal(err)
	}
	if _, err := t2.WriteContentsTo(buf); err != nil {
		t.Fatal(err)
	}

	t3, err := ReadTensor(t1.DataType(), t1.Shape(), buf)
	if err != nil {
		t.Fatal(err)
	}
	t4, err := ReadTensor(t2.DataType(), t2.Shape(), buf)
	if err != nil {
		t.Fatal(err)
	}

	if v, ok := t3.Value().(int8); !ok || v != 7 {
		t.Errorf("Got (%v (%T), %v), want (7 (int8), true)", v, v, ok)
	}
	if v, ok := t4.Value().(float32); !ok || v != 2.718 {
		t.Errorf("Got (%v (%T), %v), want (2.718 (float32), true)", v, v, ok)
	}
}

func TestTensorSerializationErrors(t *testing.T) {
	// String tensors cannot be serialized
	t1, err := NewTensor("abcd")
	if err != nil {
		t.Fatal(err)
	}
	buf := new(bytes.Buffer)
	if n, err := t1.WriteContentsTo(buf); n != 0 || err == nil || buf.Len() != 0 {
		t.Errorf("Got (%v, %v, %v) want (0, <non-nil>, 0)", n, err, buf.Len())
	}
	// Should fail to read a truncated value.
	if t1, err = NewTensor(int8(8)); err != nil {
		t.Fatal(err)
	}
	n, err := t1.WriteContentsTo(buf)
	if err != nil {
		t.Fatal(err)
	}
	r := bytes.NewReader(buf.Bytes()[:n-1])
	if _, err = ReadTensor(t1.DataType(), t1.Shape(), r); err == nil {
		t.Error("ReadTensor should have failed if the tensor content was truncated")
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
