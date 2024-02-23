// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.12
// +build go1.12

package impl

import "reflect"

func mapRange(v reflect.Value) *reflect.MapIter { return v.MapRange() }
