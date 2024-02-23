// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl_test

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

func TestExtensionType(t *testing.T) {
	cmpOpts := cmp.Options{
		cmp.Comparer(func(x, y proto.Message) bool {
			return proto.Equal(x, y)
		}),
	}
	for _, test := range []struct {
		xt    protoreflect.ExtensionType
		value interface{}
	}{
		{
			xt:    testpb.E_OptionalInt32,
			value: int32(0),
		},
		{
			xt:    testpb.E_OptionalInt64,
			value: int64(0),
		},
		{
			xt:    testpb.E_OptionalUint32,
			value: uint32(0),
		},
		{
			xt:    testpb.E_OptionalUint64,
			value: uint64(0),
		},
		{
			xt:    testpb.E_OptionalFloat,
			value: float32(0),
		},
		{
			xt:    testpb.E_OptionalDouble,
			value: float64(0),
		},
		{
			xt:    testpb.E_OptionalBool,
			value: true,
		},
		{
			xt:    testpb.E_OptionalString,
			value: "",
		},
		{
			xt:    testpb.E_OptionalBytes,
			value: []byte{},
		},
		{
			xt:    testpb.E_OptionalNestedMessage,
			value: &testpb.TestAllExtensions_NestedMessage{},
		},
		{
			xt:    testpb.E_OptionalNestedEnum,
			value: testpb.TestAllTypes_FOO,
		},
		{
			xt:    testpb.E_RepeatedInt32,
			value: []int32{0},
		},
		{
			xt:    testpb.E_RepeatedInt64,
			value: []int64{0},
		},
		{
			xt:    testpb.E_RepeatedUint32,
			value: []uint32{0},
		},
		{
			xt:    testpb.E_RepeatedUint64,
			value: []uint64{0},
		},
		{
			xt:    testpb.E_RepeatedFloat,
			value: []float32{0},
		},
		{
			xt:    testpb.E_RepeatedDouble,
			value: []float64{0},
		},
		{
			xt:    testpb.E_RepeatedBool,
			value: []bool{true},
		},
		{
			xt:    testpb.E_RepeatedString,
			value: []string{""},
		},
		{
			xt:    testpb.E_RepeatedBytes,
			value: [][]byte{nil},
		},
		{
			xt:    testpb.E_RepeatedNestedMessage,
			value: []*testpb.TestAllExtensions_NestedMessage{{}},
		},
		{
			xt:    testpb.E_RepeatedNestedEnum,
			value: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO},
		},
	} {
		name := test.xt.TypeDescriptor().FullName()
		t.Run(fmt.Sprint(name), func(t *testing.T) {
			if !test.xt.IsValidInterface(test.value) {
				t.Fatalf("IsValidInterface(%[1]T(%[1]v)) = false, want true", test.value)
			}
			v := test.xt.ValueOf(test.value)
			if !test.xt.IsValidValue(v) {
				t.Fatalf("IsValidValue(%[1]T(%[1]v)) = false, want true", v)
			}
			if got, want := test.xt.InterfaceOf(v), test.value; !cmp.Equal(got, want, cmpOpts) {
				t.Fatalf("round trip InterfaceOf(ValueOf(x)) = %v, want %v", got, want)
			}
		})
	}
}
