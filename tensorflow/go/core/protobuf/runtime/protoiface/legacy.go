// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoiface

type MessageV1 interface {
	Reset()
	String() string
	ProtoMessage()
}

type ExtensionRangeV1 struct {
	Start, End int32 // both inclusive
}
