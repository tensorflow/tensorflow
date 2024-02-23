// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego && !appengine
// +build !purego,!appengine

package impl

// When using unsafe pointers, we can just treat enum values as int32s.

var (
	coderEnumNoZero      = coderInt32NoZero
	coderEnum            = coderInt32
	coderEnumPtr         = coderInt32Ptr
	coderEnumSlice       = coderInt32Slice
	coderEnumPackedSlice = coderInt32PackedSlice
)
