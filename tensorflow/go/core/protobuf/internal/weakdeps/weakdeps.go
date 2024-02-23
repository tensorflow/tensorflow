// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build weak_dependency
// +build weak_dependency

package weakdeps

import (
	// Ensure that any program using "github.com/golang/protobuf"
	// uses a version that wraps this module so that there is a
	// unified view on what protobuf types are globally registered.
	_ "github.com/golang/protobuf/proto"
)
