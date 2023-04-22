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
	"reflect"
	"testing"
)

func TestShape(t *testing.T) {
	tests := []struct {
		shape Shape
		slice []int64
		full  bool
		str   string
	}{
		{
			shape: ScalarShape(),
			slice: make([]int64, 0),
			full:  true,
			str:   "[]",
		},
		{
			shape: MakeShape(-1, 2, -1, 4),
			slice: []int64{-1, 2, -1, 4},
			full:  false,
			str:   "[?, 2, ?, 4]",
		},
		{
			shape: MakeShape(2, 3),
			slice: []int64{2, 3},
			full:  true,
			str:   "[2, 3]",
		},
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("%#v", test.shape), func(t *testing.T) {
			if got, want := test.shape.NumDimensions(), len(test.slice); got != want {
				t.Errorf("Got %v, want %v", got, want)
			}
			if gotSlice, err := test.shape.ToSlice(); err != nil || !reflect.DeepEqual(gotSlice, test.slice) {
				t.Errorf("Got (%#v, %v), want (%#v, nil)", gotSlice, err, test.slice)
			}
			if got, want := test.shape.IsFullySpecified(), test.full; got != want {
				t.Errorf("Got %v, want %v", got, want)
			}
			if got, want := test.shape.String(), test.str; got != want {
				t.Errorf("Got %v, want %v", got, want)
			}
		})
	}

}

func TestZeroShape(t *testing.T) {
	var s Shape
	if s.NumDimensions() != -1 {
		t.Error(s.NumDimensions())
	}
	if _, err := s.ToSlice(); err == nil {
		t.Error("ToSlice() on a Shape of unknown number of dimensions should fail")
	}
	if s.IsFullySpecified() {
		t.Error("Shape of unknown number of dimensions should not be fully specified")
	}
	if got, want := s.String(), "?"; got != want {
		t.Errorf("Got %q, want %q", got, want)
	}

}
