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
	"io"
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
		{nil, uint32(5)},
		{nil, uint64(5)},
		{nil, float32(5)},
		{nil, float64(5)},
		{nil, complex(float32(5), float32(6))},
		{nil, complex(float64(5), float64(6))},
		{nil, "a string"},
		{[]int64{1}, []uint32{1}},
		{[]int64{1}, []uint64{1}},
		{[]int64{2}, []bool{true, false}},
		{[]int64{1}, []float64{1}},
		{[]int64{1}, [1]float64{1}},
		{[]int64{1, 1}, [1][1]float64{{1}}},
		{[]int64{1, 1, 1}, [1][1][]float64{{{1}}}},
		{[]int64{1, 1, 2}, [1][][2]float64{{{1, 2}}}},
		{[]int64{1, 1, 1, 1}, [1][][1][]float64{{{{1}}}}},
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
		// Mismatched dimensions
		[][]float32{{1, 2, 3}, {4}},
		// Mismatched dimensions. Should return "mismatched slice lengths" error instead of "BUG"
		[][][]float32{{{1, 2}, {3, 4}}, {{1}, {3}}},
		// Mismatched dimensions. Should return error instead of valid tensor
		[][][]float32{{{1, 2}, {3, 4}}, {{1}, {3}}, {{1, 2, 3}, {2, 3, 4}}},
		// Mismatched dimensions for strings
		[][]string{{"abc"}, {"abcd", "abcd"}},
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

func TestReadTensorReadAll(t *testing.T) {
	// Get the bytes of a tensor.
	a := []float32{1.1, 1.2, 1.3}
	ats, err := NewTensor(a)
	if err != nil {
		t.Fatal(err)
	}
	abuf := new(bytes.Buffer)
	if _, err := ats.WriteContentsTo(abuf); err != nil {
		t.Fatal(err)
	}

	// Get the bytes of another tensor.
	b := []float32{1.1, 1.2, 1.3}
	bts, err := NewTensor(b)
	if err != nil {
		t.Fatal(err)
	}
	bbuf := new(bytes.Buffer)
	if _, err := bts.WriteContentsTo(bbuf); err != nil {
		t.Fatal(err)
	}

	// Check that ReadTensor reads all bytes of both tensors, when the situation
	// requires one than reads.
	abbuf := io.MultiReader(abuf, bbuf)
	abts, err := ReadTensor(Float, []int64{2, 3}, abbuf)
	if err != nil {
		t.Fatal(err)
	}
	abtsf32 := abts.Value().([][]float32)
	expected := [][]float32{a, b}

	if len(abtsf32) != 2 {
		t.Fatalf("first dimension %d is not 2", len(abtsf32))
	}
	for i := 0; i < 2; i++ {
		if len(abtsf32[i]) != 3 {
			t.Fatalf("second dimension %d is not 3", len(abtsf32[i]))
		}
		for j := 0; j < 3; j++ {
			if abtsf32[i][j] != expected[i][j] {
				t.Errorf("value at %d %d not equal %f %f", i, j, abtsf32[i][j], expected[i][j])
			}
		}
	}
}

func TestReadTensorNegativeDimention(t *testing.T) {
	buf := new(bytes.Buffer)
	_, err := ReadTensor(Int32, []int64{-1, 1}, buf)
	if err == nil {
		t.Fatal("ReadTensor should failed if shape contains negative dimention")
	}
}

func benchmarkNewTensor(b *testing.B, v interface{}) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if t, err := NewTensor(v); err != nil || t == nil {
			b.Fatalf("(%v, %v)", t, err)
		}
	}
}

func benchmarkValueTensor(b *testing.B, v interface{}) {
	t, err := NewTensor(v)
	if err != nil {
		b.Fatalf("(%v, %v)", t, err)
	}
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_ = t.Value()
	}
}

func BenchmarkTensor(b *testing.B) {
	// Some sample sizes from the Inception image labeling model.
	// Where input tensors correspond to a 224x224 RGB image
	// flattened into a vector.
	var vector [224 * 224 * 3]int32
	var arrays [100][100][100]int32

	l3 := make([][][]float32, 100)
	l2 := make([][]float32, 100*100)
	l1 := make([]float32, 100*100*100)
	for i := range l2 {
		l2[i] = l1[i*100 : (i+1)*100]
	}
	for i := range l3 {
		l3[i] = l2[i*100 : (i+1)*100]
	}

	s1 := make([]string, 100*100*100)
	s2 := make([][]string, 100*100)
	s3 := make([][][]string, 100)
	for i := range s1 {
		s1[i] = "cheesit"
	}
	for i := range s2 {
		s2[i] = s1[i*100 : (i+1)*100]
	}
	for i := range s3 {
		s3[i] = s2[i*100 : (i+1)*100]
	}

	tests := []interface{}{
		vector,
		arrays,
		l1,
		l2,
		l3,
		s1,
		s2,
		s3,
	}
	b.Run("New", func(b *testing.B) {
		for _, test := range tests {
			b.Run(fmt.Sprintf("%T", test), func(b *testing.B) { benchmarkNewTensor(b, test) })
		}
	})
	b.Run("Value", func(b *testing.B) {
		for _, test := range tests {
			b.Run(fmt.Sprintf("%T", test), func(b *testing.B) { benchmarkValueTensor(b, test) })
		}
	})

}

func TestReshape(t *testing.T) {
	tensor, err := NewTensor([]int64{1, 2})
	if err != nil {
		t.Fatalf("Unable to create new tensor: %v", err)
	}

	if got, want := len(tensor.Shape()), 1; got != want {
		t.Fatalf("len(tensor.Shape()): got %d, want %d", got, want)
	}
	if got, want := tensor.Shape()[0], int64(2); got != want {
		t.Errorf("tensor.Shape()[0]: got %d, want %d", got, want)
	}

	if err := tensor.Reshape([]int64{1, 2}); err != nil {
		t.Fatalf("tensor.Reshape([1, 2]) failed: %v", err)
	}

	if got, want := len(tensor.Shape()), 2; got != want {
		t.Fatalf("After reshape, len(tensor.Shape()): got %d, want %d", got, want)
	}
	if got, want := tensor.Shape()[0], int64(1); got != want {
		t.Errorf("After reshape, tensor.Shape()[0]: got %d, want %d", got, want)
	}
	if got, want := tensor.Shape()[1], int64(2); got != want {
		t.Errorf("After reshape, tensor.Shape()[1]: got %d, want %d", got, want)
	}
}
