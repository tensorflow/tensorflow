// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"reflect"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// Equal reports whether two messages are equal,
// by recursively comparing the fields of the message.
//
//   - Bytes fields are equal if they contain identical bytes.
//     Empty bytes (regardless of nil-ness) are considered equal.
//
//   - Floating-point fields are equal if they contain the same value.
//     Unlike the == operator, a NaN is equal to another NaN.
//
//   - Other scalar fields are equal if they contain the same value.
//
//   - Message fields are equal if they have
//     the same set of populated known and extension field values, and
//     the same set of unknown fields values.
//
//   - Lists are equal if they are the same length and
//     each corresponding element is equal.
//
//   - Maps are equal if they have the same set of keys and
//     the corresponding value for each key is equal.
//
// An invalid message is not equal to a valid message.
// An invalid message is only equal to another invalid message of the
// same type. An invalid message often corresponds to a nil pointer
// of the concrete message type. For example, (*pb.M)(nil) is not equal
// to &pb.M{}.
// If two valid messages marshal to the same bytes under deterministic
// serialization, then Equal is guaranteed to report true.
func Equal(x, y Message) bool {
	if x == nil || y == nil {
		return x == nil && y == nil
	}
	if reflect.TypeOf(x).Kind() == reflect.Ptr && x == y {
		// Avoid an expensive comparison if both inputs are identical pointers.
		return true
	}
	mx := x.ProtoReflect()
	my := y.ProtoReflect()
	if mx.IsValid() != my.IsValid() {
		return false
	}
	vx := protoreflect.ValueOfMessage(mx)
	vy := protoreflect.ValueOfMessage(my)
	return vx.Equal(vy)
}
