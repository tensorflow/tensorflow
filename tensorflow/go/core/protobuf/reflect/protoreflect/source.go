// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import (
	"strconv"
)

// SourceLocations is a list of source locations.
type SourceLocations interface {
	// Len reports the number of source locations in the proto file.
	Len() int
	// Get returns the ith SourceLocation. It panics if out of bounds.
	Get(int) SourceLocation

	// ByPath returns the SourceLocation for the given path,
	// returning the first location if multiple exist for the same path.
	// If multiple locations exist for the same path,
	// then SourceLocation.Next index can be used to identify the
	// index of the next SourceLocation.
	// If no location exists for this path, it returns the zero value.
	ByPath(path SourcePath) SourceLocation

	// ByDescriptor returns the SourceLocation for the given descriptor,
	// returning the first location if multiple exist for the same path.
	// If no location exists for this descriptor, it returns the zero value.
	ByDescriptor(desc Descriptor) SourceLocation

	doNotImplement
}

// SourceLocation describes a source location and
// corresponds with the google.protobuf.SourceCodeInfo.Location message.
type SourceLocation struct {
	// Path is the path to the declaration from the root file descriptor.
	// The contents of this slice must not be mutated.
	Path SourcePath

	// StartLine and StartColumn are the zero-indexed starting location
	// in the source file for the declaration.
	StartLine, StartColumn int
	// EndLine and EndColumn are the zero-indexed ending location
	// in the source file for the declaration.
	// In the descriptor.proto, the end line may be omitted if it is identical
	// to the start line. Here, it is always populated.
	EndLine, EndColumn int

	// LeadingDetachedComments are the leading detached comments
	// for the declaration. The contents of this slice must not be mutated.
	LeadingDetachedComments []string
	// LeadingComments is the leading attached comment for the declaration.
	LeadingComments string
	// TrailingComments is the trailing attached comment for the declaration.
	TrailingComments string

	// Next is an index into SourceLocations for the next source location that
	// has the same Path. It is zero if there is no next location.
	Next int
}

// SourcePath identifies part of a file descriptor for a source location.
// The SourcePath is a sequence of either field numbers or indexes into
// a repeated field that form a path starting from the root file descriptor.
//
// See google.protobuf.SourceCodeInfo.Location.path.
type SourcePath []int32

// Equal reports whether p1 equals p2.
func (p1 SourcePath) Equal(p2 SourcePath) bool {
	if len(p1) != len(p2) {
		return false
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			return false
		}
	}
	return true
}

// String formats the path in a humanly readable manner.
// The output is guaranteed to be deterministic,
// making it suitable for use as a key into a Go map.
// It is not guaranteed to be stable as the exact output could change
// in a future version of this module.
//
// Example output:
//
//	.message_type[6].nested_type[15].field[3]
func (p SourcePath) String() string {
	b := p.appendFileDescriptorProto(nil)
	for _, i := range p {
		b = append(b, '.')
		b = strconv.AppendInt(b, int64(i), 10)
	}
	return string(b)
}

type appendFunc func(*SourcePath, []byte) []byte

func (p *SourcePath) appendSingularField(b []byte, name string, f appendFunc) []byte {
	if len(*p) == 0 {
		return b
	}
	b = append(b, '.')
	b = append(b, name...)
	*p = (*p)[1:]
	if f != nil {
		b = f(p, b)
	}
	return b
}

func (p *SourcePath) appendRepeatedField(b []byte, name string, f appendFunc) []byte {
	b = p.appendSingularField(b, name, nil)
	if len(*p) == 0 || (*p)[0] < 0 {
		return b
	}
	b = append(b, '[')
	b = strconv.AppendUint(b, uint64((*p)[0]), 10)
	b = append(b, ']')
	*p = (*p)[1:]
	if f != nil {
		b = f(p, b)
	}
	return b
}
