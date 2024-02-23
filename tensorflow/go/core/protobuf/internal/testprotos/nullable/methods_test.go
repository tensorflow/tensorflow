// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// For messages which do not provide legacy Marshal and Unmarshal methods,
// only test compatibility with the Marshal/Unmarshal functionality with
// pure protobuf reflection since there is no support for nullable fields
// in the table-driven implementation.
//go:build protoreflect
// +build protoreflect

package nullable

import "google.golang.org/protobuf/runtime/protoimpl"

func init() {
	methodTestProtos = append(methodTestProtos,
		protoimpl.X.ProtoMessageV2Of((*Proto2)(nil)).ProtoReflect().Type(),
		protoimpl.X.ProtoMessageV2Of((*Proto3)(nil)).ProtoReflect().Type(),
	)
}
