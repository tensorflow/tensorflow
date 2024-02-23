// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package pragma provides types that can be embedded into a struct to
// statically enforce or prevent certain language properties.
package pragma

import "sync"

// NoUnkeyedLiterals can be embedded in a struct to prevent unkeyed literals.
type NoUnkeyedLiterals struct{}

// DoNotImplement can be embedded in an interface to prevent trivial
// implementations of the interface.
//
// This is useful to prevent unauthorized implementations of an interface
// so that it can be extended in the future for any protobuf language changes.
type DoNotImplement interface{ ProtoInternal(DoNotImplement) }

// DoNotCompare can be embedded in a struct to prevent comparability.
type DoNotCompare [0]func()

// DoNotCopy can be embedded in a struct to help prevent shallow copies.
// This does not rely on a Go language feature, but rather a special case
// within the vet checker.
//
// See https://golang.org/issues/8005.
type DoNotCopy [0]sync.Mutex
