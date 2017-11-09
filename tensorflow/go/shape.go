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
	"strings"
)

// Shape represents the (possibly partially known) shape of a tensor that will
// be produced by an operation.
//
// The zero-value of a Shape represents a shape with an unknown number of
// dimensions.
type Shape struct {
	dims []int64
}

// ScalarShape returns a Shape representing a scalar.
func ScalarShape() Shape {
	return Shape{dims: make([]int64, 0)}
}

// MakeShape returns a Shape with the provided size of each dimension.
//
// A value of -1 implies that the size of the corresponding dimension is not
// known.
func MakeShape(shape ...int64) Shape {
	cpy := make([]int64, len(shape))
	copy(cpy, shape)
	return Shape{dims: cpy}
}

// NumDimensions returns the number of dimensions represented by s, or -1 if
// unknown.
func (s Shape) NumDimensions() int {
	if s.dims == nil {
		return -1
	}
	return len(s.dims)
}

// Size returns the size of the dim-th dimension of the shape, or -1 if it
// is unknown.
//
// REQUIRES: 0 <= dim < s.NumDimensions()
func (s Shape) Size(dim int) int64 {
	if dim < 0 || dim >= s.NumDimensions() {
		return -1
	}
	return s.dims[dim]
}

// IsFullySpecified returns true iff the size of all the dimensions of s are
// known.
func (s Shape) IsFullySpecified() bool {
	if s.dims == nil {
		return false
	}
	for _, size := range s.dims {
		if size <= 1 {
			return false
		}
	}
	return true
}

// ToSlice returns the (possibly partially known) shape represented by s as a
// slice, or an error if the number of dimensions is not known.
func (s Shape) ToSlice() ([]int64, error) {
	if s.dims == nil {
		return nil, fmt.Errorf("cannot create a slice for a Shape with an unknown number of dimensions")
	}
	cpy := make([]int64, len(s.dims))
	copy(cpy, s.dims)
	return cpy, nil
}

func (s Shape) String() string {
	if s.dims == nil {
		return "?"
	}
	ret := fmt.Sprint(s.dims)
	for _, size := range s.dims {
		if size < 0 {
			ret = strings.Replace(ret, fmt.Sprint(size), "?", 1)
		}
	}
	return strings.Replace(ret, " ", ", ", -1)
}
