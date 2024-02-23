// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protopath provides functionality for
// representing a sequence of protobuf reflection operations on a message.
package protopath

import (
	"fmt"

	"google.golang.org/protobuf/internal/msgfmt"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// NOTE: The Path and Values are separate types here since there are use cases
// where you would like to "address" some value in a message with just the path
// and don't have the value information available.
//
// This is different from how github.com/google/go-cmp/cmp.Path operates,
// which combines both path and value information together.
// Since the cmp package itself is the only one ever constructing a cmp.Path,
// it will always have the value available.

// Path is a sequence of protobuf reflection steps applied to some root
// protobuf message value to arrive at the current value.
// The first step must be a [Root] step.
type Path []Step

// TODO: Provide a Parse function that parses something similar to or
// perhaps identical to the output of Path.String.

// Index returns the ith step in the path and supports negative indexing.
// A negative index starts counting from the tail of the Path such that -1
// refers to the last step, -2 refers to the second-to-last step, and so on.
// It returns a zero Step value if the index is out-of-bounds.
func (p Path) Index(i int) Step {
	if i < 0 {
		i = len(p) + i
	}
	if i < 0 || i >= len(p) {
		return Step{}
	}
	return p[i]
}

// String returns a structured representation of the path
// by concatenating the string representation of every path step.
func (p Path) String() string {
	var b []byte
	for _, s := range p {
		b = s.appendString(b)
	}
	return string(b)
}

// Values is a Path paired with a sequence of values at each step.
// The lengths of [Values.Path] and [Values.Values] must be identical.
// The first step must be a [Root] step and
// the first value must be a concrete message value.
type Values struct {
	Path   Path
	Values []protoreflect.Value
}

// Len reports the length of the path and values.
// If the path and values have differing length, it returns the minimum length.
func (p Values) Len() int {
	n := len(p.Path)
	if n > len(p.Values) {
		n = len(p.Values)
	}
	return n
}

// Index returns the ith step and value and supports negative indexing.
// A negative index starts counting from the tail of the Values such that -1
// refers to the last pair, -2 refers to the second-to-last pair, and so on.
func (p Values) Index(i int) (out struct {
	Step  Step
	Value protoreflect.Value
}) {
	// NOTE: This returns a single struct instead of two return values so that
	// callers can make use of the the value in an expression:
	//	vs.Index(i).Value.Interface()
	n := p.Len()
	if i < 0 {
		i = n + i
	}
	if i < 0 || i >= n {
		return out
	}
	out.Step = p.Path[i]
	out.Value = p.Values[i]
	return out
}

// String returns a humanly readable representation of the path and last value.
// Do not depend on the output being stable.
//
// For example:
//
//	(path.to.MyMessage).list_field[5].map_field["hello"] = {hello: "world"}
func (p Values) String() string {
	n := p.Len()
	if n == 0 {
		return ""
	}

	// Determine the field descriptor associated with the last step.
	var fd protoreflect.FieldDescriptor
	last := p.Index(-1)
	switch last.Step.kind {
	case FieldAccessStep:
		fd = last.Step.FieldDescriptor()
	case MapIndexStep, ListIndexStep:
		fd = p.Index(-2).Step.FieldDescriptor()
	}

	// Format the full path with the last value.
	return fmt.Sprintf("%v = %v", p.Path[:n], msgfmt.FormatValue(last.Value, fd))
}
