// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protojson marshals and unmarshals protocol buffer messages as JSON
// format. It follows the guide at
// https://protobuf.dev/programming-guides/proto3#json.
//
// This package produces a different output than the standard [encoding/json]
// package, which does not operate correctly on protocol buffer messages.
package protojson
