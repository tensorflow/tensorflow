// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocmp

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protopack"
	"google.golang.org/protobuf/types/dynamicpb"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

func TestEqual(t *testing.T) {
	type test struct {
		x, y interface{}
		opts cmp.Options
		want bool
	}
	var tests []test

	allTypesDesc := (*testpb.TestAllTypes)(nil).ProtoReflect().Descriptor()

	// Test nil and empty messages of differing types.
	tests = append(tests, []test{{
		x:    (*testpb.TestAllTypes)(nil),
		y:    (*testpb.TestAllTypes)(nil),
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    (*testpb.TestAllTypes)(nil),
		y:    (*testpb.TestAllExtensions)(nil),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    (*testpb.TestAllTypes)(nil),
		y:    new(testpb.TestAllTypes),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    (*testpb.TestAllTypes)(nil),
		y:    dynamicpb.NewMessage(allTypesDesc),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    (*testpb.TestAllTypes)(nil),
		y:    new(testpb.TestAllTypes),
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    (*testpb.TestAllTypes)(nil),
		y:    dynamicpb.NewMessage(allTypesDesc),
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    new(testpb.TestAllTypes),
		y:    new(testpb.TestAllTypes),
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    new(testpb.TestAllTypes),
		y:    dynamicpb.NewMessage(allTypesDesc),
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    new(testpb.TestAllTypes),
		y:    new(testpb.TestAllExtensions),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ I interface{} }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ I interface{} }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ I interface{} }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    struct{ I interface{} }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ I interface{} }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    struct{ I interface{} }{new(testpb.TestAllTypes)},
		y:    struct{ I interface{} }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ I interface{} }{new(testpb.TestAllTypes)},
		y:    struct{ I interface{} }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ M proto.Message }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ M proto.Message }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ M proto.Message }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    struct{ M proto.Message }{(*testpb.TestAllTypes)(nil)},
		y:    struct{ M proto.Message }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    struct{ M proto.Message }{new(testpb.TestAllTypes)},
		y:    struct{ M proto.Message }{new(testpb.TestAllTypes)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ M proto.Message }{new(testpb.TestAllTypes)},
		y:    struct{ M proto.Message }{dynamicpb.NewMessage(allTypesDesc)},
		opts: cmp.Options{Transform()},
		want: true,
	}}...)

	// Test message values.
	tests = append(tests, []test{{
		x:    testpb.TestAllTypes{OptionalSint64: proto.Int64(1)},
		y:    testpb.TestAllTypes{OptionalSint64: proto.Int64(1)},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    testpb.TestAllTypes{OptionalSint64: proto.Int64(1)},
		y:    testpb.TestAllTypes{OptionalSint64: proto.Int64(2)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ M testpb.TestAllTypes }{M: testpb.TestAllTypes{OptionalSint64: proto.Int64(1)}},
		y:    struct{ M testpb.TestAllTypes }{M: testpb.TestAllTypes{OptionalSint64: proto.Int64(1)}},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ M testpb.TestAllTypes }{M: testpb.TestAllTypes{OptionalSint64: proto.Int64(1)}},
		y:    struct{ M testpb.TestAllTypes }{M: testpb.TestAllTypes{OptionalSint64: proto.Int64(2)}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    struct{ M []testpb.TestAllTypes }{M: []testpb.TestAllTypes{{OptionalSint64: proto.Int64(1)}}},
		y:    struct{ M []testpb.TestAllTypes }{M: []testpb.TestAllTypes{{OptionalSint64: proto.Int64(1)}}},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    struct{ M []testpb.TestAllTypes }{M: []testpb.TestAllTypes{{OptionalSint64: proto.Int64(1)}}},
		y:    struct{ M []testpb.TestAllTypes }{M: []testpb.TestAllTypes{{OptionalSint64: proto.Int64(2)}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: struct {
			M map[string]testpb.TestAllTypes
		}{
			M: map[string]testpb.TestAllTypes{"k": {OptionalSint64: proto.Int64(1)}},
		},
		y: struct {
			M map[string]testpb.TestAllTypes
		}{
			M: map[string]testpb.TestAllTypes{"k": {OptionalSint64: proto.Int64(1)}},
		},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x: struct {
			M map[string]testpb.TestAllTypes
		}{
			M: map[string]testpb.TestAllTypes{"k": {OptionalSint64: proto.Int64(1)}},
		},
		y: struct {
			M map[string]testpb.TestAllTypes
		}{
			M: map[string]testpb.TestAllTypes{"k": {OptionalSint64: proto.Int64(2)}},
		},
		opts: cmp.Options{Transform()},
		want: false,
	}}...)

	// Test IgnoreUnknown.
	raw := protopack.Message{
		protopack.Tag{1, protopack.BytesType}, protopack.String("Hello, goodbye!"),
	}.Marshal()
	tests = append(tests, []test{{
		x:    apply(&testpb.TestAllTypes{OptionalSint64: proto.Int64(5)}, setUnknown{raw}),
		y:    &testpb.TestAllTypes{OptionalSint64: proto.Int64(5)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    apply(&testpb.TestAllTypes{OptionalSint64: proto.Int64(5)}, setUnknown{raw}),
		y:    &testpb.TestAllTypes{OptionalSint64: proto.Int64(5)},
		opts: cmp.Options{Transform(), IgnoreUnknown()},
		want: true,
	}, {
		x:    apply(&testpb.TestAllTypes{OptionalSint64: proto.Int64(5)}, setUnknown{raw}),
		y:    &testpb.TestAllTypes{OptionalSint64: proto.Int64(6)},
		opts: cmp.Options{Transform(), IgnoreUnknown()},
		want: false,
	}, {
		x:    apply(&testpb.TestAllTypes{OptionalSint64: proto.Int64(5)}, setUnknown{raw}),
		y:    apply(dynamicpb.NewMessage(allTypesDesc), setField{6, int64(5)}),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    apply(&testpb.TestAllTypes{OptionalSint64: proto.Int64(5)}, setUnknown{raw}),
		y:    apply(dynamicpb.NewMessage(allTypesDesc), setField{6, int64(5)}),
		opts: cmp.Options{Transform(), IgnoreUnknown()},
		want: true,
	}}...)

	// Test IgnoreDefaultScalars.
	tests = append(tests, []test{{
		x: &testpb.TestAllTypes{
			DefaultInt32:  proto.Int32(81),
			DefaultUint32: proto.Uint32(83),
			DefaultFloat:  proto.Float32(91.5),
			DefaultBool:   proto.Bool(true),
			DefaultBytes:  []byte("world"),
		},
		y: &testpb.TestAllTypes{
			DefaultInt64:       proto.Int64(82),
			DefaultUint64:      proto.Uint64(84),
			DefaultDouble:      proto.Float64(92e3),
			DefaultString:      proto.String("hello"),
			DefaultForeignEnum: testpb.ForeignEnum_FOREIGN_BAR.Enum(),
		},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{
			DefaultInt32:  proto.Int32(81),
			DefaultUint32: proto.Uint32(83),
			DefaultFloat:  proto.Float32(91.5),
			DefaultBool:   proto.Bool(true),
			DefaultBytes:  []byte("world"),
		},
		y: &testpb.TestAllTypes{
			DefaultInt64:       proto.Int64(82),
			DefaultUint64:      proto.Uint64(84),
			DefaultDouble:      proto.Float64(92e3),
			DefaultString:      proto.String("hello"),
			DefaultForeignEnum: testpb.ForeignEnum_FOREIGN_BAR.Enum(),
		},
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: true,
	}, {
		x: &testpb.TestAllTypes{
			OptionalInt32:  proto.Int32(81),
			OptionalUint32: proto.Uint32(83),
			OptionalFloat:  proto.Float32(91.5),
			OptionalBool:   proto.Bool(true),
			OptionalBytes:  []byte("world"),
		},
		y: &testpb.TestAllTypes{
			OptionalInt64:       proto.Int64(82),
			OptionalUint64:      proto.Uint64(84),
			OptionalDouble:      proto.Float64(92e3),
			OptionalString:      proto.String("hello"),
			OptionalForeignEnum: testpb.ForeignEnum_FOREIGN_BAR.Enum(),
		},
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{
			OptionalInt32:  proto.Int32(0),
			OptionalUint32: proto.Uint32(0),
			OptionalFloat:  proto.Float32(0),
			OptionalBool:   proto.Bool(false),
			OptionalBytes:  []byte(""),
		},
		y: &testpb.TestAllTypes{
			OptionalInt64:       proto.Int64(0),
			OptionalUint64:      proto.Uint64(0),
			OptionalDouble:      proto.Float64(0),
			OptionalString:      proto.String(""),
			OptionalForeignEnum: testpb.ForeignEnum_FOREIGN_FOO.Enum(),
		},
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: true,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_DefaultInt32, int32(81)},
			setExtension{testpb.E_DefaultUint32, uint32(83)},
			setExtension{testpb.E_DefaultFloat, float32(91.5)},
			setExtension{testpb.E_DefaultBool, bool(true)},
			setExtension{testpb.E_DefaultBytes, []byte("world")}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_DefaultInt64, int64(82)},
			setExtension{testpb.E_DefaultUint64, uint64(84)},
			setExtension{testpb.E_DefaultDouble, float64(92e3)},
			setExtension{testpb.E_DefaultString, string("hello")}),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_DefaultInt32, int32(81)},
			setExtension{testpb.E_DefaultUint32, uint32(83)},
			setExtension{testpb.E_DefaultFloat, float32(91.5)},
			setExtension{testpb.E_DefaultBool, bool(true)},
			setExtension{testpb.E_DefaultBytes, []byte("world")}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_DefaultInt64, int64(82)},
			setExtension{testpb.E_DefaultUint64, uint64(84)},
			setExtension{testpb.E_DefaultDouble, float64(92e3)},
			setExtension{testpb.E_DefaultString, string("hello")}),
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: true,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalInt32, int32(0)},
			setExtension{testpb.E_OptionalUint32, uint32(0)},
			setExtension{testpb.E_OptionalFloat, float32(0)},
			setExtension{testpb.E_OptionalBool, bool(false)},
			setExtension{testpb.E_OptionalBytes, []byte("")}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalInt64, int64(0)},
			setExtension{testpb.E_OptionalUint64, uint64(0)},
			setExtension{testpb.E_OptionalDouble, float64(0)},
			setExtension{testpb.E_OptionalString, string("")}),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalInt32, int32(0)},
			setExtension{testpb.E_OptionalUint32, uint32(0)},
			setExtension{testpb.E_OptionalFloat, float32(0)},
			setExtension{testpb.E_OptionalBool, bool(false)},
			setExtension{testpb.E_OptionalBytes, []byte("")}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalInt64, int64(0)},
			setExtension{testpb.E_OptionalUint64, uint64(0)},
			setExtension{testpb.E_OptionalDouble, float64(0)},
			setExtension{testpb.E_OptionalString, string("")}),
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: true,
	}, {
		x: &testpb.TestAllTypes{
			DefaultFloat: proto.Float32(91.6),
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{
			OptionalForeignMessage: &testpb.ForeignMessage{},
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(), IgnoreDefaultScalars()},
		want: false,
	}}...)

	// Test IgnoreEmptyMessages.
	tests = append(tests, []test{{
		x:    []*testpb.TestAllTypes{nil, {}, {OptionalInt32: proto.Int32(5)}},
		y:    []*testpb.TestAllTypes{nil, {}, {OptionalInt32: proto.Int32(5)}},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    []*testpb.TestAllTypes{nil, {}, {OptionalInt32: proto.Int32(5)}},
		y:    []*testpb.TestAllTypes{{OptionalInt32: proto.Int32(5)}},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{OptionalForeignMessage: &testpb.ForeignMessage{}},
		y:    &testpb.TestAllTypes{OptionalForeignMessage: nil},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{OptionalForeignMessage: &testpb.ForeignMessage{}},
		y:    &testpb.TestAllTypes{OptionalForeignMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{OptionalForeignMessage: &testpb.ForeignMessage{C: proto.Int32(5)}},
		y:    &testpb.TestAllTypes{OptionalForeignMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: nil},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: nil},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(5)}, {}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(5)}, {}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {}, nil, {}, {C: proto.Int32(5)}, {}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(5)}, {}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {}, nil, {}, {C: proto.Int32(5)}, {}}},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: nil},
		opts: cmp.Options{Transform()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": nil, "2": {}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: nil},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": nil, "2": {}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": nil, "2": {A: proto.Int32(5)}, "3": {}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: nil},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": nil, "2": {A: proto.Int32(5)}, "3": {}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": {}, "1a": {}, "1b": nil, "2": {A: proto.Int32(5)}, "4": {}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": nil, "2": {A: proto.Int32(5)}, "3": {}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"1": {}, "1a": {}, "1b": nil, "2": {A: proto.Int32(5)}, "4": {}}},
		opts: cmp.Options{Transform(), IgnoreEmptyMessages()},
		want: true,
	}}...)

	// Test IgnoreEnums and IgnoreMessages.
	tests = append(tests, []test{{
		x: &testpb.TestAllTypes{
			OptionalNestedMessage:  &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)},
			RepeatedNestedMessage:  []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}},
			MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"3": {A: proto.Int32(3)}},
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{
			OptionalNestedMessage:  &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)},
			RepeatedNestedMessage:  []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}},
			MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"3": {A: proto.Int32(3)}},
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(), IgnoreMessages(&testpb.TestAllTypes{})},
		want: true,
	}, {
		x: &testpb.TestAllTypes{
			OptionalNestedEnum:  testpb.TestAllTypes_FOO.Enum(),
			RepeatedNestedEnum:  []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR},
			MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"baz": testpb.TestAllTypes_BAZ},
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{
			OptionalNestedEnum:  testpb.TestAllTypes_FOO.Enum(),
			RepeatedNestedEnum:  []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR},
			MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"baz": testpb.TestAllTypes_BAZ},
		},
		y:    &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(), IgnoreEnums(testpb.TestAllTypes_NestedEnum(0))},
		want: true,
	}, {
		x: &testpb.TestAllTypes{
			OptionalNestedEnum:  testpb.TestAllTypes_FOO.Enum(),
			RepeatedNestedEnum:  []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR},
			MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"baz": testpb.TestAllTypes_BAZ},

			OptionalNestedMessage:  &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)},
			RepeatedNestedMessage:  []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}},
			MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"3": {A: proto.Int32(3)}},
		},
		y: &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(),
			IgnoreMessages(&testpb.TestAllExtensions{}),
			IgnoreEnums(testpb.ForeignEnum(0)),
		},
		want: false,
	}}...)

	// Test IgnoreFields and IgnoreOneofs.
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{OptionalInt32: proto.Int32(5)},
		y:    &testpb.TestAllTypes{OptionalInt32: proto.Int32(6)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(5)},
		y: &testpb.TestAllTypes{},
		opts: cmp.Options{Transform(),
			IgnoreFields(&testpb.TestAllTypes{}, "optional_int32")},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(5)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(6)},
		opts: cmp.Options{Transform(),
			IgnoreFields(&testpb.TestAllTypes{}, "optional_int32")},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(5)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(6)},
		opts: cmp.Options{Transform(),
			IgnoreFields(&testpb.TestAllTypes{}, "optional_int64")},
		want: false,
	}, {
		x:    &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{5}},
		y:    &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofString{"5"}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{5}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofString{"5"}},
		opts: cmp.Options{Transform(),
			IgnoreFields(&testpb.TestAllTypes{}, "oneof_uint32"),
			IgnoreFields(&testpb.TestAllTypes{}, "oneof_string")},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{5}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofString{"5"}},
		opts: cmp.Options{Transform(),
			IgnoreOneofs(&testpb.TestAllTypes{}, "oneof_field")},
		want: true,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "hello"}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "goodbye"}),
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "hello"}),
		y: new(testpb.TestAllExtensions),
		opts: cmp.Options{Transform(),
			IgnoreDescriptors(testpb.E_OptionalString.TypeDescriptor())},
		want: true,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "hello"}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "goodbye"}),
		opts: cmp.Options{Transform(),
			IgnoreDescriptors(testpb.E_OptionalString.TypeDescriptor())},
		want: true,
	}, {
		x: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "hello"}),
		y: apply(new(testpb.TestAllExtensions),
			setExtension{testpb.E_OptionalString, "goodbye"}),
		opts: cmp.Options{Transform(),
			IgnoreDescriptors(testpb.E_OptionalInt32.TypeDescriptor())},
		want: false,
	}}...)

	// Test FilterEnum.
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y:    &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.ForeignEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y testpb.TestAllTypes_NestedEnum) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_FOO.Enum()},
		y: &testpb.TestAllTypes{OptionalNestedEnum: testpb.TestAllTypes_BAR.Enum()},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y Enum) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.ForeignEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y []testpb.TestAllTypes_NestedEnum) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y []Enum) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y:    &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.ForeignEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y map[string]testpb.TestAllTypes_NestedEnum) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_FOO}},
		y: &testpb.TestAllTypes{MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{"k": testpb.TestAllTypes_BAR}},
		opts: cmp.Options{
			Transform(),
			FilterEnum(testpb.TestAllTypes_NestedEnum(0), cmp.Comparer(func(x, y map[string]Enum) bool { return true })),
		},
		want: true,
	}}...)

	// Test FilterMessage.
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y:    &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllExtensions), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y *testpb.TestAllTypes_NestedMessage) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(1)}},
		y: &testpb.TestAllTypes{OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(2)}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y Message) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y:    &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllExtensions), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y []*testpb.TestAllTypes_NestedMessage) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y []Message) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y:    &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllExtensions), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y int) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y map[string]*testpb.TestAllTypes_NestedMessage) bool { return true })),
		},
		want: false, // matching filter type, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(1)}}},
		y: &testpb.TestAllTypes{MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k": {A: proto.Int32(2)}}},
		opts: cmp.Options{
			Transform(),
			FilterMessage(new(testpb.TestAllTypes_NestedMessage), cmp.Comparer(func(x, y map[string]Message) bool { return true })),
		},
		want: true,
	}}...)

	// Test FilterField.
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
		y:    &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "optional_int64", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter name
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "optional_int32", cmp.Comparer(func(x, y int64) bool { return true })),
		},
		want: false, // matching filter name, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "optional_int32", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OptionalInt32: proto.Int32(1)},
		y: &testpb.TestAllTypes{OptionalInt32: proto.Int32(2)},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "optional_int32", cmp.Comparer(func(x, y int32) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
		y:    &testpb.TestAllTypes{RepeatedInt32: []int32{2}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "repeated_int64", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter name
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "repeated_int32", cmp.Comparer(func(x, y []int64) bool { return true })),
		},
		want: false, // matching filter name, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "repeated_int32", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{1}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "repeated_int32", cmp.Comparer(func(x, y []int32) bool { return true })),
		},
		want: true,
	}, {
		x:    &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 1}},
		y:    &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{2: 2}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 1}},
		y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{2: 2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "map_int64_int64", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter name
	}, {
		x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 1}},
		y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{2: 2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "map_int32_int32", cmp.Comparer(func(x, y map[int64]int64) bool { return true })),
		},
		want: false, // matching filter name, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 1}},
		y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{2: 2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "map_int32_int32", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{1: 1}},
		y: &testpb.TestAllTypes{MapInt32Int32: map[int32]int32{2: 2}},
		opts: cmp.Options{
			Transform(),
			FilterField(new(testpb.TestAllTypes), "map_int32_int32", cmp.Comparer(func(x, y map[int32]int32) bool { return true })),
		},
		want: true,
	}}...)

	// Test FilterOneof
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{1}},
		y:    &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{2}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{1}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{2}},
		opts: cmp.Options{
			Transform(),
			FilterOneof(new(testpb.TestAllTypes), "oneof_optional", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: false, // mismatching filter name
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{1}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{2}},
		opts: cmp.Options{
			Transform(),
			FilterOneof(new(testpb.TestAllTypes), "oneof_field", cmp.Comparer(func(x, y string) bool { return true })),
		},
		want: false, // matching filter name, but mismatching comparer type
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{1}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{2}},
		opts: cmp.Options{
			Transform(),
			FilterOneof(new(testpb.TestAllTypes), "oneof_field", cmp.Comparer(func(x, y uint32) bool { return true })),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{1}},
		y: &testpb.TestAllTypes{OneofField: &testpb.TestAllTypes_OneofUint32{2}},
		opts: cmp.Options{
			Transform(),
			FilterOneof(new(testpb.TestAllTypes), "oneof_field", cmp.Comparer(func(x, y interface{}) bool { return true })),
		},
		want: true,
	}}...)

	// Test SortRepeated.
	type higherOrderType struct {
		M    *testpb.TestAllTypes
		I32s []int32
		Es   []testpb.TestAllTypes_NestedEnum
		Ms   []*testpb.ForeignMessage
	}
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y:    &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y int32) bool { return x < y }),
		},
		want: true,
	}, {
		x: higherOrderType{
			M:    &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
			I32s: []int32{3, 2, 1, 2, 3, 3},
		},
		y: higherOrderType{
			M:    &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
			I32s: []int32{2, 3, 3, 2, 1, 3},
		},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y int32) bool { return x < y }),
		},
		want: false, // sort does not apply to []int32 outside of a message
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y int64) bool { return x < y }),
		},
		want: false, // wrong sort type: int32 != int64
	}, {
		x:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y testpb.TestAllTypes_NestedEnum) bool { return x < y }),
		},
		want: true,
	}, {
		x: higherOrderType{
			M:  &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
			Es: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ},
		},
		y: higherOrderType{
			M:  &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
			Es: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ},
		},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y testpb.TestAllTypes_NestedEnum) bool { return x < y }),
		},
		want: false, // sort does not apply to []testpb.TestAllTypes_NestedEnum outside of a message
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y testpb.ForeignEnum) bool { return x < y }),
		},
		want: false, // wrong sort type: testpb.TestAllTypes_NestedEnum != testpb.ForeignEnum
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y *testpb.ForeignMessage) bool { return x.GetC() < y.GetC() }),
		},
		want: true,
	}, {
		x: higherOrderType{
			M:  &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
			Ms: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}},
		},
		y: higherOrderType{
			M:  &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
			Ms: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}},
		},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y *testpb.ForeignMessage) bool { return x.GetC() < y.GetC() }),
		},
		want: false, // sort does not apply to []*testpb.ForeignMessage outside of a message
	}, {
		x: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y *testpb.TestAllTypes_NestedMessage) bool { return x.GetA() < y.GetA() }),
		},
		want: false, // wrong sort type: *testpb.ForeignMessage != *testpb.TestAllTypes_NestedMessage
	}, {
		x: &testpb.TestAllTypes{
			RepeatedInt32:    []int32{-32, +32},
			RepeatedSint32:   []int32{-32, +32},
			RepeatedSfixed32: []int32{-32, +32},
			RepeatedInt64:    []int64{-64, +64},
			RepeatedSint64:   []int64{-64, +64},
			RepeatedSfixed64: []int64{-64, +64},
			RepeatedUint32:   []uint32{0, 32},
			RepeatedFixed32:  []uint32{0, 32},
			RepeatedUint64:   []uint64{0, 64},
			RepeatedFixed64:  []uint64{0, 64},
		},
		y: &testpb.TestAllTypes{
			RepeatedInt32:    []int32{+32, -32},
			RepeatedSint32:   []int32{+32, -32},
			RepeatedSfixed32: []int32{+32, -32},
			RepeatedInt64:    []int64{+64, -64},
			RepeatedSint64:   []int64{+64, -64},
			RepeatedSfixed64: []int64{+64, -64},
			RepeatedUint32:   []uint32{32, 0},
			RepeatedFixed32:  []uint32{32, 0},
			RepeatedUint64:   []uint64{64, 0},
			RepeatedFixed64:  []uint64{64, 0},
		},
		opts: cmp.Options{
			Transform(),
			SortRepeated(func(x, y int32) bool { return x < y }),
			SortRepeated(func(x, y int64) bool { return x < y }),
			SortRepeated(func(x, y uint32) bool { return x < y }),
			SortRepeated(func(x, y uint64) bool { return x < y }),
		},
		want: true,
	}}...)

	// Test SortRepeatedFields.
	tests = append(tests, []test{{
		x:    &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y:    &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_int32"),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedInt32: []int32{3, 2, 1, 2, 3, 3}},
		y: &testpb.TestAllTypes{RepeatedInt32: []int32{2, 3, 3, 2, 1, 3}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_int64"),
		},
		want: false, // wrong field: repeated_int32 != repeated_int64
	}, {
		x:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y:    &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_nested_enum"),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAR, testpb.TestAllTypes_BAZ}},
		y: &testpb.TestAllTypes{RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_BAR, testpb.TestAllTypes_FOO, testpb.TestAllTypes_BAZ}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_foreign_enum"),
		},
		want: false, // wrong field: repeated_nested_enum != repeated_foreign_enum
	}, {
		x:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y:    &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{Transform()},
		want: false,
	}, {
		x: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_foreign_message"),
		},
		want: true,
	}, {
		x: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{{}, {C: proto.Int32(3)}, nil, {C: proto.Int32(3)}, {C: proto.Int32(5)}, {C: proto.Int32(4)}}},
		y: &testpb.TestAllTypes{RepeatedForeignMessage: []*testpb.ForeignMessage{nil, {C: proto.Int32(3)}, {}, {C: proto.Int32(4)}, {C: proto.Int32(3)}, {C: proto.Int32(5)}}},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes), "repeated_nested_message"),
		},
		want: false, // wrong field: repeated_foreign_message != repeated_nested_message
	}, {
		x: &testpb.TestAllTypes{
			RepeatedBool:           []bool{false, true},
			RepeatedInt32:          []int32{-32, +32},
			RepeatedInt64:          []int64{-64, +64},
			RepeatedUint32:         []uint32{0, 32},
			RepeatedUint64:         []uint64{0, 64},
			RepeatedFloat:          []float32{-32.32, +32.32},
			RepeatedDouble:         []float64{-64.64, +64.64},
			RepeatedString:         []string{"hello", "world"},
			RepeatedBytes:          [][]byte{[]byte("hello"), []byte("world")},
			RepeatedForeignEnum:    []testpb.ForeignEnum{testpb.ForeignEnum_FOREIGN_FOO, testpb.ForeignEnum_FOREIGN_BAR},
			RepeatedForeignMessage: []*testpb.ForeignMessage{{C: proto.Int32(-1)}, {C: proto.Int32(+1)}},
		},
		y: &testpb.TestAllTypes{
			RepeatedBool:           []bool{true, false},
			RepeatedInt32:          []int32{+32, -32},
			RepeatedInt64:          []int64{+64, -64},
			RepeatedUint32:         []uint32{32, 0},
			RepeatedUint64:         []uint64{64, 0},
			RepeatedFloat:          []float32{+32.32, -32.32},
			RepeatedDouble:         []float64{+64.64, -64.64},
			RepeatedString:         []string{"world", "hello"},
			RepeatedBytes:          [][]byte{[]byte("world"), []byte("hello")},
			RepeatedForeignEnum:    []testpb.ForeignEnum{testpb.ForeignEnum_FOREIGN_BAR, testpb.ForeignEnum_FOREIGN_FOO},
			RepeatedForeignMessage: []*testpb.ForeignMessage{{C: proto.Int32(+1)}, {C: proto.Int32(-1)}},
		},
		opts: cmp.Options{
			Transform(),
			SortRepeatedFields(new(testpb.TestAllTypes),
				"repeated_bool",
				"repeated_int32",
				"repeated_int64",
				"repeated_uint32",
				"repeated_uint64",
				"repeated_float",
				"repeated_double",
				"repeated_string",
				"repeated_bytes",
				"repeated_foreign_enum",
				"repeated_foreign_message",
			),
		},
		want: true,
	}}...)

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := cmp.Equal(tt.x, tt.y, tt.opts)
			if got != tt.want {
				if !got {
					t.Errorf("cmp.Equal = false, want true; diff:\n%v", cmp.Diff(tt.x, tt.y, tt.opts))
				} else {
					t.Errorf("cmp.Equal = true, want false")
				}
			}
		})
	}
}

type setField struct {
	num protoreflect.FieldNumber
	val interface{}
}
type setUnknown struct {
	raw protoreflect.RawFields
}
type setExtension struct {
	typ protoreflect.ExtensionType
	val interface{}
}

// apply applies a sequence of mutating operations to m.
func apply(m proto.Message, ops ...interface{}) proto.Message {
	mr := m.ProtoReflect()
	md := mr.Descriptor()
	for _, op := range ops {
		switch op := op.(type) {
		case setField:
			fd := md.Fields().ByNumber(op.num)
			mr.Set(fd, protoreflect.ValueOf(op.val))
		case setUnknown:
			mr.SetUnknown(op.raw)
		case setExtension:
			mr.Set(op.typ.TypeDescriptor(), protoreflect.ValueOf(op.val))
		}
	}
	return m
}

func TestSort(t *testing.T) {
	t.Run("F32", func(t *testing.T) {
		want := []float32{
			float32(math.Float32frombits(0xffc00000)), // -NaN
			float32(math.Inf(-1)),
			float32(-math.MaxFloat32),
			float32(-123.456),
			float32(-math.SmallestNonzeroFloat32),
			float32(math.Copysign(0, -1)),
			float32(math.Copysign(0, +1)),
			float32(+math.SmallestNonzeroFloat32),
			float32(+123.456),
			float32(+math.MaxFloat32),
			float32(math.Inf(+1)),
			float32(math.Float32frombits(0x7fc00000)), // +NaN
		}
		for i := 0; i < 10; i++ {
			t.Run("", func(t *testing.T) {
				got := append([]float32(nil), want...)
				rn := rand.New(rand.NewSource(int64(i)))
				for i, j := range rn.Perm(len(got)) {
					got[i], got[j] = got[j], got[i]
				}
				sort.Slice(got, func(i, j int) bool {
					return lessF32(got[i], got[j])
				})
				cmpF32s := cmp.Comparer(func(x, y float32) bool {
					return math.Float32bits(x) == math.Float32bits(y)
				})
				if diff := cmp.Diff(want, got, cmpF32s); diff != "" {
					t.Errorf("Sort mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})
	t.Run("F64", func(t *testing.T) {
		want := []float64{
			float64(math.Float64frombits(0xfff8000000000001)), // -NaN
			float64(math.Inf(-1)),
			float64(-math.MaxFloat64),
			float64(-123.456),
			float64(-math.SmallestNonzeroFloat64),
			float64(math.Copysign(0, -1)),
			float64(math.Copysign(0, +1)),
			float64(+math.SmallestNonzeroFloat64),
			float64(+123.456),
			float64(+math.MaxFloat64),
			float64(math.Inf(+1)),
			float64(math.Float64frombits(0x7ff8000000000001)), // +NaN
		}
		for i := 0; i < 10; i++ {
			t.Run("", func(t *testing.T) {
				got := append([]float64(nil), want...)
				rn := rand.New(rand.NewSource(int64(i)))
				for i, j := range rn.Perm(len(got)) {
					got[i], got[j] = got[j], got[i]
				}
				sort.Slice(got, func(i, j int) bool {
					return lessF64(got[i], got[j])
				})
				cmpF64s := cmp.Comparer(func(x, y float64) bool {
					return math.Float64bits(x) == math.Float64bits(y)
				})
				if diff := cmp.Diff(want, got, cmpF64s); diff != "" {
					t.Errorf("Sort mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})
}
