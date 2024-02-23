// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"math"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protopack"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	test3pb "google.golang.org/protobuf/internal/testprotos/test3"
	testeditionspb "google.golang.org/protobuf/internal/testprotos/testeditions"
)

func TestEqual(t *testing.T) {
	identicalPtrPb := &testpb.TestAllTypes{MapStringString: map[string]string{"a": "b", "c": "d"}}

	type incomparableMessage struct {
		*testpb.TestAllTypes
		pragma.DoNotCompare
	}

	tests := []struct {
		x, y proto.Message
		eq   bool
	}{
		{
			x:  nil,
			y:  nil,
			eq: true,
		}, {
			x:  (*testpb.TestAllTypes)(nil),
			y:  nil,
			eq: false,
		}, {
			x:  (*testpb.TestAllTypes)(nil),
			y:  (*testpb.TestAllTypes)(nil),
			eq: true,
		}, {
			x:  new(testpb.TestAllTypes),
			y:  (*testpb.TestAllTypes)(nil),
			eq: false,
		}, {
			x:  new(testpb.TestAllTypes),
			y:  new(testpb.TestAllTypes),
			eq: true,
		}, {
			x:  (*testpb.TestAllTypes)(nil),
			y:  (*testpb.TestAllExtensions)(nil),
			eq: false,
		}, {
			x:  (*testpb.TestAllTypes)(nil),
			y:  new(testpb.TestAllExtensions),
			eq: false,
		}, {
			x:  new(testpb.TestAllTypes),
			y:  new(testpb.TestAllExtensions),
			eq: false,
		},

		// Identical input pointers
		{
			x:  identicalPtrPb,
			y:  identicalPtrPb,
			eq: true,
		},

		// Incomparable types. The top-level types are not actually directly
		// compared (which would panic), but rather the comparison happens on the
		// objects returned by ProtoReflect(). These tests are here just to ensure
		// that any short-circuit checks do not accidentally try to compare
		// incomparable top-level types.
		{
			x:  incomparableMessage{TestAllTypes: identicalPtrPb},
			y:  incomparableMessage{TestAllTypes: identicalPtrPb},
			eq: true,
		},
		{
			x:  identicalPtrPb,
			y:  incomparableMessage{TestAllTypes: identicalPtrPb},
			eq: true,
		},
		{
			x:  identicalPtrPb,
			y:  &incomparableMessage{TestAllTypes: identicalPtrPb},
			eq: true,
		},

		// Proto2 scalars.
		{
			x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
			y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalInt64: proto.Int64(1)},
			y: &testpb.TestAllTypes{OptionalInt64: proto.Int64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalUint32: proto.Uint32(1)},
			y: &testpb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalUint64: proto.Uint64(1)},
			y: &testpb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalSint32: proto.Int32(1)},
			y: &testpb.TestAllTypes{OptionalSint32: proto.Int32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalSint64: proto.Int64(1)},
			y: &testpb.TestAllTypes{OptionalSint64: proto.Int64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalFixed32: proto.Uint32(1)},
			y: &testpb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalFixed64: proto.Uint64(1)},
			y: &testpb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalSfixed32: proto.Int32(1)},
			y: &testpb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalSfixed64: proto.Int64(1)},
			y: &testpb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalFloat: proto.Float32(1)},
			y: &testpb.TestAllTypes{OptionalFloat: proto.Float32(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalDouble: proto.Float64(1)},
			y: &testpb.TestAllTypes{OptionalDouble: proto.Float64(2)},
		}, {
			x: &testpb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			y: &testpb.TestAllTypes{OptionalFloat: proto.Float32(0)},
		}, {
			x: &testpb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			y: &testpb.TestAllTypes{OptionalDouble: proto.Float64(0)},
		}, {
			x: &testpb.TestAllTypes{OptionalBool: proto.Bool(true)},
			y: &testpb.TestAllTypes{OptionalBool: proto.Bool(false)},
		}, {
			x: &testpb.TestAllTypes{OptionalString: proto.String("a")},
			y: &testpb.TestAllTypes{OptionalString: proto.String("b")},
		}, {
			x: &testpb.TestAllTypes{OptionalBytes: []byte("a")},
			y: &testpb.TestAllTypes{OptionalBytes: []byte("b")},
		}, {
			x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
			y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		}, {
			x:  &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
			y:  &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalInt64: proto.Int64(2)},
			y:  &testpb.TestAllTypes{OptionalInt64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
			y:  &testpb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
			y:  &testpb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalSint32: proto.Int32(2)},
			y:  &testpb.TestAllTypes{OptionalSint32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalSint64: proto.Int64(2)},
			y:  &testpb.TestAllTypes{OptionalSint64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
			y:  &testpb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
			y:  &testpb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
			y:  &testpb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
			y:  &testpb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalFloat: proto.Float32(2)},
			y:  &testpb.TestAllTypes{OptionalFloat: proto.Float32(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalDouble: proto.Float64(2)},
			y:  &testpb.TestAllTypes{OptionalDouble: proto.Float64(2)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			y:  &testpb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			y:  &testpb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalBool: proto.Bool(true)},
			y:  &testpb.TestAllTypes{OptionalBool: proto.Bool(true)},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalString: proto.String("abc")},
			y:  &testpb.TestAllTypes{OptionalString: proto.String("abc")},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalBytes: []byte("abc")},
			y:  &testpb.TestAllTypes{OptionalBytes: []byte("abc")},
			eq: true,
		}, {
			x:  &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
			y:  &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
			eq: true,
		},

		// Editions scalars.
		{
			x: &testeditionspb.TestAllTypes{OptionalInt32: proto.Int32(1)},
			y: &testeditionspb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalInt64: proto.Int64(1)},
			y: &testeditionspb.TestAllTypes{OptionalInt64: proto.Int64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalUint32: proto.Uint32(1)},
			y: &testeditionspb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalUint64: proto.Uint64(1)},
			y: &testeditionspb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalSint32: proto.Int32(1)},
			y: &testeditionspb.TestAllTypes{OptionalSint32: proto.Int32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalSint64: proto.Int64(1)},
			y: &testeditionspb.TestAllTypes{OptionalSint64: proto.Int64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalFixed32: proto.Uint32(1)},
			y: &testeditionspb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalFixed64: proto.Uint64(1)},
			y: &testeditionspb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalSfixed32: proto.Int32(1)},
			y: &testeditionspb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalSfixed64: proto.Int64(1)},
			y: &testeditionspb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(1)},
			y: &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(1)},
			y: &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(2)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			y: &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			y: &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalBool: proto.Bool(true)},
			y: &testeditionspb.TestAllTypes{OptionalBool: proto.Bool(false)},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalString: proto.String("a")},
			y: &testeditionspb.TestAllTypes{OptionalString: proto.String("b")},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalBytes: []byte("a")},
			y: &testeditionspb.TestAllTypes{OptionalBytes: []byte("b")},
		}, {
			x: &testeditionspb.TestAllTypes{OptionalNestedEnum: testeditionspb.TestAllTypes_FOO.Enum()},
			y: &testeditionspb.TestAllTypes{OptionalNestedEnum: testeditionspb.TestAllTypes_BAR.Enum()},
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalInt32: proto.Int32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalInt32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalInt64: proto.Int64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalInt64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalUint32: proto.Uint32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalUint64: proto.Uint64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalSint32: proto.Int32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalSint32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalSint64: proto.Int64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalSint64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalFixed32: proto.Uint32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalFixed64: proto.Uint64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalSfixed32: proto.Int32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalSfixed64: proto.Int64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(2)},
			y:  &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(2)},
			y:  &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(2)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			y:  &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(float32(math.NaN()))},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			y:  &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(float64(math.NaN()))},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalBool: proto.Bool(true)},
			y:  &testeditionspb.TestAllTypes{OptionalBool: proto.Bool(true)},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalString: proto.String("abc")},
			y:  &testeditionspb.TestAllTypes{OptionalString: proto.String("abc")},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalBytes: []byte("abc")},
			y:  &testeditionspb.TestAllTypes{OptionalBytes: []byte("abc")},
			eq: true,
		}, {
			x:  &testeditionspb.TestAllTypes{OptionalNestedEnum: testeditionspb.TestAllTypes_FOO.Enum()},
			y:  &testeditionspb.TestAllTypes{OptionalNestedEnum: testeditionspb.TestAllTypes_FOO.Enum()},
			eq: true,
		},

		// Proto2 presence.
		{
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalInt64: proto.Int64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalUint32: proto.Uint32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalUint64: proto.Uint64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalSint32: proto.Int32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalSint64: proto.Int64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalFixed32: proto.Uint32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalFixed64: proto.Uint64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalSfixed32: proto.Int32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalSfixed64: proto.Int64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalFloat: proto.Float32(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalDouble: proto.Float64(0)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalBool: proto.Bool(false)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalString: proto.String("")},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalBytes: []byte{}},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		},

		// Editions presence.
		{
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalInt32: proto.Int32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalInt64: proto.Int64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalUint32: proto.Uint32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalUint64: proto.Uint64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalSint32: proto.Int32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalSint64: proto.Int64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalFixed32: proto.Uint32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalFixed64: proto.Uint64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalSfixed32: proto.Int32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalSfixed64: proto.Int64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalFloat: proto.Float32(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalDouble: proto.Float64(0)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalBool: proto.Bool(false)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalString: proto.String("")},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalBytes: []byte{}},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalNestedEnum: testeditionspb.TestAllTypes_FOO.Enum()},
		},

		// Proto3 presence.
		{
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalInt32: proto.Int32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalInt64: proto.Int64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalUint32: proto.Uint32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalUint64: proto.Uint64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalSint32: proto.Int32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalSint64: proto.Int64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalFixed32: proto.Uint32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalFixed64: proto.Uint64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalSfixed32: proto.Int32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalSfixed64: proto.Int64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalFloat: proto.Float32(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalDouble: proto.Float64(0)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalBool: proto.Bool(false)},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalString: proto.String("")},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalBytes: []byte{}},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalNestedEnum: test3pb.TestAllTypes_FOO.Enum()},
		},

		// Proto2 default values are not considered by Equal, so the following are still unequal.
		{
			x: &testpb.TestAllTypes{DefaultInt32: proto.Int32(81)},
			y: &testpb.TestAllTypes{},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultInt32: proto.Int32(81)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultInt64: proto.Int64(82)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultUint32: proto.Uint32(83)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultUint64: proto.Uint64(84)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultSint32: proto.Int32(-85)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultSint64: proto.Int64(86)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultFixed32: proto.Uint32(87)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultFixed64: proto.Uint64(88)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultSfixed32: proto.Int32(89)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultSfixed64: proto.Int64(-90)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultFloat: proto.Float32(91.5)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultDouble: proto.Float64(92e3)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultBool: proto.Bool(true)},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultString: proto.String("hello")},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultBytes: []byte("world")},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{DefaultNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		},

		// Edition default values are not considered by Equal, so the following are still unequal.
		{
			x: &testeditionspb.TestAllTypes{DefaultInt32: proto.Int32(81)},
			y: &testeditionspb.TestAllTypes{},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultInt32: proto.Int32(81)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultInt64: proto.Int64(82)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultUint32: proto.Uint32(83)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultUint64: proto.Uint64(84)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultSint32: proto.Int32(-85)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultSint64: proto.Int64(86)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultFixed32: proto.Uint32(87)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultFixed64: proto.Uint64(88)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultSfixed32: proto.Int32(89)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultSfixed64: proto.Int64(-90)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultFloat: proto.Float32(91.5)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultDouble: proto.Float64(92e3)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultBool: proto.Bool(true)},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultString: proto.String("hello")},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultBytes: []byte("world")},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{DefaultNestedEnum: testeditionspb.TestAllTypes_BAR.Enum()},
		},

		// Groups.
		{
			x: &testpb.TestAllTypes{Optionalgroup: &testpb.TestAllTypes_OptionalGroup{
				A: proto.Int32(1),
			}},
			y: &testpb.TestAllTypes{Optionalgroup: &testpb.TestAllTypes_OptionalGroup{
				A: proto.Int32(2),
			}},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{Optionalgroup: &testpb.TestAllTypes_OptionalGroup{}},
		}, {
			x: &testeditionspb.TestAllTypes{Optionalgroup: &testeditionspb.TestAllTypes_OptionalGroup{
				A: proto.Int32(1),
			}},
			y: &testeditionspb.TestAllTypes{Optionalgroup: &testeditionspb.TestAllTypes_OptionalGroup{
				A: proto.Int32(2),
			}},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{Optionalgroup: &testeditionspb.TestAllTypes_OptionalGroup{}},
		},

		// Messages.
		{
			x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
				A: proto.Int32(1),
			}},
			y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
				A: proto.Int32(2),
			}},
		}, {
			x: &testpb.TestAllTypes{},
			y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{}},
		}, {
			x: &testeditionspb.TestAllTypes{},
			y: &testeditionspb.TestAllTypes{OptionalNestedMessage: &testeditionspb.TestAllTypes_NestedMessage{}},
		}, {
			x: &test3pb.TestAllTypes{},
			y: &test3pb.TestAllTypes{OptionalNestedMessage: &test3pb.TestAllTypes_NestedMessage{}},
		},

		// Lists.
		{
			x: &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
			y: &testpb.TestAllTypes{RepeatedInt32: []int32{1, 2}},
		}, {
			x: &testpb.TestAllTypes{RepeatedInt32: []int32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedInt32: []int32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedInt64: []int64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedInt64: []int64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedUint32: []uint32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedUint32: []uint32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedUint64: []uint64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedUint64: []uint64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedSint32: []int32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedSint32: []int32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedSint64: []int64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedSint64: []int64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedFixed32: []uint32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedFixed32: []uint32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedFixed64: []uint64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedFixed64: []uint64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedSfixed32: []int32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedSfixed32: []int32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedSfixed64: []int64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedSfixed64: []int64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedFloat: []float32{1, 2}},
			y: &testpb.TestAllTypes{RepeatedFloat: []float32{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedDouble: []float64{1, 2}},
			y: &testpb.TestAllTypes{RepeatedDouble: []float64{1, 3}},
		}, {
			x: &testpb.TestAllTypes{RepeatedBool: []bool{true, false}},
			y: &testpb.TestAllTypes{RepeatedBool: []bool{true, true}},
		}, {
			x: &testpb.TestAllTypes{RepeatedString: []string{"a", "b"}},
			y: &testpb.TestAllTypes{RepeatedString: []string{"a", "c"}},
		}, {
			x: &testpb.TestAllTypes{RepeatedBytes: [][]byte{[]byte("a"), []byte("b")}},
			y: &testpb.TestAllTypes{RepeatedBytes: [][]byte{[]byte("a"), []byte("c")}},
		}, {
			x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
			y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		}, {
			x: &testpb.TestAllTypes{Repeatedgroup: []*testpb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1)},
				{A: proto.Int32(2)},
			}},
			y: &testpb.TestAllTypes{Repeatedgroup: []*testpb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1)},
				{A: proto.Int32(3)},
			}},
		}, {
			x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				{A: proto.Int32(2)},
			}},
			y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				{A: proto.Int32(3)},
			}},
		},

		// Editions Lists.
		{
			x: &testeditionspb.TestAllTypes{RepeatedInt32: []int32{1}},
			y: &testeditionspb.TestAllTypes{RepeatedInt32: []int32{1, 2}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedInt32: []int32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedInt32: []int32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedInt64: []int64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedInt64: []int64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedUint32: []uint32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedUint32: []uint32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedUint64: []uint64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedUint64: []uint64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedSint32: []int32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedSint32: []int32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedSint64: []int64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedSint64: []int64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedFixed32: []uint32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedFixed32: []uint32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedFixed64: []uint64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedFixed64: []uint64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedSfixed32: []int32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedSfixed32: []int32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedSfixed64: []int64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedSfixed64: []int64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedFloat: []float32{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedFloat: []float32{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedDouble: []float64{1, 2}},
			y: &testeditionspb.TestAllTypes{RepeatedDouble: []float64{1, 3}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedBool: []bool{true, false}},
			y: &testeditionspb.TestAllTypes{RepeatedBool: []bool{true, true}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedString: []string{"a", "b"}},
			y: &testeditionspb.TestAllTypes{RepeatedString: []string{"a", "c"}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedBytes: [][]byte{[]byte("a"), []byte("b")}},
			y: &testeditionspb.TestAllTypes{RepeatedBytes: [][]byte{[]byte("a"), []byte("c")}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedNestedEnum: []testeditionspb.TestAllTypes_NestedEnum{testeditionspb.TestAllTypes_FOO}},
			y: &testeditionspb.TestAllTypes{RepeatedNestedEnum: []testeditionspb.TestAllTypes_NestedEnum{testeditionspb.TestAllTypes_BAR}},
		}, {
			x: &testeditionspb.TestAllTypes{Repeatedgroup: []*testeditionspb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1)},
				{A: proto.Int32(2)},
			}},
			y: &testeditionspb.TestAllTypes{Repeatedgroup: []*testeditionspb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1)},
				{A: proto.Int32(3)},
			}},
		}, {
			x: &testeditionspb.TestAllTypes{RepeatedNestedMessage: []*testeditionspb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				{A: proto.Int32(2)},
			}},
			y: &testeditionspb.TestAllTypes{RepeatedNestedMessage: []*testeditionspb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				{A: proto.Int32(3)},
			}},
		},

		// Maps: various configurations.
		{
			x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
			y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{3: 4}},
		}, {
			x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
			y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
		}, {
			x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
			y: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{3: 4}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
			y: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2}},
		},

		// Maps: various types.
		{
			x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapInt64Int64: map[int64]int64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapInt64Int64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapUint32Uint32: map[uint32]uint32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapUint32Uint32: map[uint32]uint32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapUint64Uint64: map[uint64]uint64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapUint64Uint64: map[uint64]uint64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapSint32Sint32: map[int32]int32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapSint32Sint32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapSint64Sint64: map[int64]int64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapSint64Sint64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapFixed32Fixed32: map[uint32]uint32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapFixed32Fixed32: map[uint32]uint32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapFixed64Fixed64: map[uint64]uint64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapFixed64Fixed64: map[uint64]uint64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapSfixed32Sfixed32: map[int32]int32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapSfixed32Sfixed32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapSfixed64Sfixed64: map[int64]int64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapSfixed64Sfixed64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapInt32Float: map[int32]float32{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapInt32Float: map[int32]float32{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapInt32Double: map[int32]float64{1: 2, 3: 4}},
			y: &testpb.TestAllTypes{MapInt32Double: map[int32]float64{1: 2, 3: 5}},
		}, {
			x: &testpb.TestAllTypes{MapBoolBool: map[bool]bool{true: false, false: true}},
			y: &testpb.TestAllTypes{MapBoolBool: map[bool]bool{true: false, false: false}},
		}, {
			x: &testpb.TestAllTypes{MapStringString: map[string]string{"a": "b", "c": "d"}},
			y: &testpb.TestAllTypes{MapStringString: map[string]string{"a": "b", "c": "e"}},
		}, {
			x: &testpb.TestAllTypes{MapStringBytes: map[string][]byte{"a": []byte("b"), "c": []byte("d")}},
			y: &testpb.TestAllTypes{MapStringBytes: map[string][]byte{"a": []byte("b"), "c": []byte("e")}},
		}, {
			x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{
				"a": {A: proto.Int32(1)},
				"b": {A: proto.Int32(2)},
			}},
			y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{
				"a": {A: proto.Int32(1)},
				"b": {A: proto.Int32(3)},
			}},
		}, {
			x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{
				"a": testpb.TestAllTypes_FOO,
				"b": testpb.TestAllTypes_BAR,
			}},
			y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{
				"a": testpb.TestAllTypes_FOO,
				"b": testpb.TestAllTypes_BAZ,
			}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt64Int64: map[int64]int64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapInt64Int64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapUint32Uint32: map[uint32]uint32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapUint32Uint32: map[uint32]uint32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapUint64Uint64: map[uint64]uint64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapUint64Uint64: map[uint64]uint64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapSint32Sint32: map[int32]int32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapSint32Sint32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapSint64Sint64: map[int64]int64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapSint64Sint64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapFixed32Fixed32: map[uint32]uint32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapFixed32Fixed32: map[uint32]uint32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapFixed64Fixed64: map[uint64]uint64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapFixed64Fixed64: map[uint64]uint64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapSfixed32Sfixed32: map[int32]int32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapSfixed32Sfixed32: map[int32]int32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapSfixed64Sfixed64: map[int64]int64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapSfixed64Sfixed64: map[int64]int64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Float: map[int32]float32{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapInt32Float: map[int32]float32{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapInt32Double: map[int32]float64{1: 2, 3: 4}},
			y: &testeditionspb.TestAllTypes{MapInt32Double: map[int32]float64{1: 2, 3: 5}},
		}, {
			x: &testeditionspb.TestAllTypes{MapBoolBool: map[bool]bool{true: false, false: true}},
			y: &testeditionspb.TestAllTypes{MapBoolBool: map[bool]bool{true: false, false: false}},
		}, {
			x: &testeditionspb.TestAllTypes{MapStringString: map[string]string{"a": "b", "c": "d"}},
			y: &testeditionspb.TestAllTypes{MapStringString: map[string]string{"a": "b", "c": "e"}},
		}, {
			x: &testeditionspb.TestAllTypes{MapStringBytes: map[string][]byte{"a": []byte("b"), "c": []byte("d")}},
			y: &testeditionspb.TestAllTypes{MapStringBytes: map[string][]byte{"a": []byte("b"), "c": []byte("e")}},
		}, {
			x: &testeditionspb.TestAllTypes{MapStringNestedMessage: map[string]*testeditionspb.TestAllTypes_NestedMessage{
				"a": {A: proto.Int32(1)},
				"b": {A: proto.Int32(2)},
			}},
			y: &testeditionspb.TestAllTypes{MapStringNestedMessage: map[string]*testeditionspb.TestAllTypes_NestedMessage{
				"a": {A: proto.Int32(1)},
				"b": {A: proto.Int32(3)},
			}},
		}, {
			x: &testeditionspb.TestAllTypes{MapStringNestedEnum: map[string]testeditionspb.TestAllTypes_NestedEnum{
				"a": testeditionspb.TestAllTypes_FOO,
				"b": testeditionspb.TestAllTypes_BAR,
			}},
			y: &testeditionspb.TestAllTypes{MapStringNestedEnum: map[string]testeditionspb.TestAllTypes_NestedEnum{
				"a": testeditionspb.TestAllTypes_FOO,
				"b": testeditionspb.TestAllTypes_BAZ,
			}},
		},

		// Extensions.
		{
			x: build(&testpb.TestAllExtensions{},
				extend(testpb.E_OptionalInt32, int32(1)),
			),
			y: build(&testpb.TestAllExtensions{},
				extend(testpb.E_OptionalInt32, int32(2)),
			),
		}, {
			x: &testpb.TestAllExtensions{},
			y: build(&testpb.TestAllExtensions{},
				extend(testpb.E_OptionalInt32, int32(2)),
			),
		},

		// Unknown fields.
		{
			x: build(&testpb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
			}.Marshal())),
			y: build(&testpb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(2),
			}.Marshal())),
		}, {
			x: build(&testpb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
			}.Marshal())),
			y: &testpb.TestAllTypes{},
		}, {
			x: build(&testeditionspb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
			}.Marshal())),
			y: build(&testeditionspb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(2),
			}.Marshal())),
		}, {
			x: build(&testeditionspb.TestAllTypes{}, unknown(protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
			}.Marshal())),
			y: &testeditionspb.TestAllTypes{},
		},
	}

	for _, tt := range tests {
		if !tt.eq && !proto.Equal(tt.x, tt.x) {
			t.Errorf("Equal(x, x) = false, want true\n==== x ====\n%v", prototext.Format(tt.x))
		}
		if !tt.eq && !proto.Equal(tt.y, tt.y) {
			t.Errorf("Equal(y, y) = false, want true\n==== y ====\n%v", prototext.Format(tt.y))
		}
		if eq := proto.Equal(tt.x, tt.y); eq != tt.eq {
			t.Errorf("Equal(x, y) = %v, want %v\n==== x ====\n%v==== y ====\n%v", eq, tt.eq, prototext.Format(tt.x), prototext.Format(tt.y))
		}
	}
}

func BenchmarkEqualWithSmallEmpty(b *testing.B) {
	x := &testpb.ForeignMessage{}
	y := &testpb.ForeignMessage{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, y)
	}
}

func BenchmarkEqualWithIdenticalPtrEmpty(b *testing.B) {
	x := &testpb.ForeignMessage{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, x)
	}
}

func BenchmarkEqualWithLargeEmpty(b *testing.B) {
	x := &testpb.TestAllTypes{}
	y := &testpb.TestAllTypes{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, y)
	}
}

func makeNested(depth int) *testpb.TestAllTypes {
	if depth <= 0 {
		return nil
	}
	return &testpb.TestAllTypes{
		OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{
			Corecursive: makeNested(depth - 1),
		},
	}
}

func BenchmarkEqualWithDeeplyNestedEqual(b *testing.B) {
	x := makeNested(20)
	y := makeNested(20)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, y)
	}
}

func BenchmarkEqualWithDeeplyNestedDifferent(b *testing.B) {
	x := makeNested(20)
	y := makeNested(21)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, y)
	}
}

func BenchmarkEqualWithDeeplyNestedIdenticalPtr(b *testing.B) {
	x := makeNested(20)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		proto.Equal(x, x)
	}
}
