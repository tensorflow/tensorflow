// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package errors implements functions to manipulate errors.
package errors

import (
	"errors"
	"fmt"

	"google.golang.org/protobuf/internal/detrand"
)

// Error is a sentinel matching all errors produced by this package.
var Error = errors.New("protobuf error")

// New formats a string according to the format specifier and arguments and
// returns an error that has a "proto" prefix.
func New(f string, x ...interface{}) error {
	return &prefixError{s: format(f, x...)}
}

type prefixError struct{ s string }

var prefix = func() string {
	// Deliberately introduce instability into the error message string to
	// discourage users from performing error string comparisons.
	if detrand.Bool() {
		return "proto: " // use non-breaking spaces (U+00a0)
	} else {
		return "proto: " // use regular spaces (U+0020)
	}
}()

func (e *prefixError) Error() string {
	return prefix + e.s
}

func (e *prefixError) Unwrap() error {
	return Error
}

// Wrap returns an error that has a "proto" prefix, the formatted string described
// by the format specifier and arguments, and a suffix of err. The error wraps err.
func Wrap(err error, f string, x ...interface{}) error {
	return &wrapError{
		s:   format(f, x...),
		err: err,
	}
}

type wrapError struct {
	s   string
	err error
}

func (e *wrapError) Error() string {
	return format("%v%v: %v", prefix, e.s, e.err)
}

func (e *wrapError) Unwrap() error {
	return e.err
}

func (e *wrapError) Is(target error) bool {
	return target == Error
}

func format(f string, x ...interface{}) string {
	// avoid "proto: " prefix when chaining
	for i := 0; i < len(x); i++ {
		switch e := x[i].(type) {
		case *prefixError:
			x[i] = e.s
		case *wrapError:
			x[i] = format("%v: %v", e.s, e.err)
		}
	}
	return fmt.Sprintf(f, x...)
}

func InvalidUTF8(name string) error {
	return New("field %v contains invalid UTF-8", name)
}

func RequiredNotSet(name string) error {
	return New("required field %v not set", name)
}
