// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego && !appengine && go1.21
// +build !purego,!appengine,go1.21

package strs

import (
	"unsafe"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// UnsafeString returns an unsafe string reference of b.
// The caller must treat the input slice as immutable.
//
// WARNING: Use carefully. The returned result must not leak to the end user
// unless the input slice is provably immutable.
func UnsafeString(b []byte) string {
	return unsafe.String(unsafe.SliceData(b), len(b))
}

// UnsafeBytes returns an unsafe bytes slice reference of s.
// The caller must treat returned slice as immutable.
//
// WARNING: Use carefully. The returned result must not leak to the end user.
func UnsafeBytes(s string) []byte {
	return unsafe.Slice(unsafe.StringData(s), len(s))
}

// Builder builds a set of strings with shared lifetime.
// This differs from strings.Builder, which is for building a single string.
type Builder struct {
	buf []byte
}

// AppendFullName is equivalent to protoreflect.FullName.Append,
// but optimized for large batches where each name has a shared lifetime.
func (sb *Builder) AppendFullName(prefix protoreflect.FullName, name protoreflect.Name) protoreflect.FullName {
	n := len(prefix) + len(".") + len(name)
	if len(prefix) == 0 {
		n -= len(".")
	}
	sb.grow(n)
	sb.buf = append(sb.buf, prefix...)
	sb.buf = append(sb.buf, '.')
	sb.buf = append(sb.buf, name...)
	return protoreflect.FullName(sb.last(n))
}

// MakeString is equivalent to string(b), but optimized for large batches
// with a shared lifetime.
func (sb *Builder) MakeString(b []byte) string {
	sb.grow(len(b))
	sb.buf = append(sb.buf, b...)
	return sb.last(len(b))
}

func (sb *Builder) grow(n int) {
	if cap(sb.buf)-len(sb.buf) >= n {
		return
	}

	// Unlike strings.Builder, we do not need to copy over the contents
	// of the old buffer since our builder provides no API for
	// retrieving previously created strings.
	sb.buf = make([]byte, 0, 2*(cap(sb.buf)+n))
}

func (sb *Builder) last(n int) string {
	return UnsafeString(sb.buf[len(sb.buf)-n:])
}
