// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.13
// +build go1.13

package errors

import "errors"

// Is is errors.Is.
func Is(err, target error) bool { return errors.Is(err, target) }
