// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocmp

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/proto"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	textpb "google.golang.org/protobuf/internal/testprotos/textpb2"
	anypb "google.golang.org/protobuf/types/known/anypb"
	wrapperspb "google.golang.org/protobuf/types/known/wrapperspb"
)

func TestReflect(t *testing.T) {
	optMsg := &testpb.TestAllTypes{
		OptionalInt32:         proto.Int32(-32),
		OptionalInt64:         proto.Int64(-64),
		OptionalUint32:        proto.Uint32(32),
		OptionalUint64:        proto.Uint64(64),
		OptionalFloat:         proto.Float32(32.32),
		OptionalDouble:        proto.Float64(64.64),
		OptionalBool:          proto.Bool(true),
		OptionalString:        proto.String("string"),
		OptionalBytes:         []byte("bytes"),
		OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(-32)},
		OptionalNestedEnum:    testpb.TestAllTypes_NEG.Enum(),
	}
	repMsg := &testpb.TestAllTypes{
		RepeatedInt32:         []int32{-32, +32},
		RepeatedInt64:         []int64{-64, +64},
		RepeatedUint32:        []uint32{0, 32},
		RepeatedUint64:        []uint64{0, 64},
		RepeatedFloat:         []float32{-32.32, +32.32},
		RepeatedDouble:        []float64{-64.64, +64.64},
		RepeatedBool:          []bool{false, true},
		RepeatedString:        []string{"hello", "goodbye"},
		RepeatedBytes:         [][]byte{[]byte("hello"), []byte("goodbye")},
		RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{{A: proto.Int32(-32)}, {A: proto.Int32(+32)}},
		RepeatedNestedEnum:    []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_NEG},
	}
	mapMsg := &testpb.TestAllTypes{
		MapInt32Int32:          map[int32]int32{-1: -32, +1: +32},
		MapInt64Int64:          map[int64]int64{-1: -32, +1: +64},
		MapUint32Uint32:        map[uint32]uint32{0: 0, 1: 32},
		MapUint64Uint64:        map[uint64]uint64{0: 0, 1: 64},
		MapInt32Float:          map[int32]float32{-1: -32.32, +1: +32.32},
		MapInt32Double:         map[int32]float64{-1: -64.64, +1: +64.64},
		MapBoolBool:            map[bool]bool{false: true, true: false},
		MapStringString:        map[string]string{"k1": "v1", "k2": "v2"},
		MapStringBytes:         map[string][]byte{"k1": []byte("v1"), "k2": []byte("v2")},
		MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{"k1": {A: proto.Int32(-32)}, "k2": {A: proto.Int32(+32)}},
		MapStringNestedEnum:    map[string]testpb.TestAllTypes_NestedEnum{"k1": testpb.TestAllTypes_FOO, "k2": testpb.TestAllTypes_NEG},
	}

	tests := []proto.Message{
		optMsg,
		repMsg,
		mapMsg,
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofUint32{32},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofUint64{64},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofFloat{32.32},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofDouble{64.64},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofBool{true},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofString{"string"},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofBytes{[]byte("bytes")},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofNestedMessage{&testpb.TestAllTypes_NestedMessage{A: proto.Int32(-32)}},
		},
		&testpb.TestAllTypes{
			OneofField: &testpb.TestAllTypes_OneofEnum{testpb.TestAllTypes_NEG},
		},
		func() proto.Message {
			m := new(testpb.TestAllExtensions)
			proto.SetExtension(m, testpb.E_OptionalInt32, int32(-32))
			proto.SetExtension(m, testpb.E_OptionalInt64, int64(-64))
			proto.SetExtension(m, testpb.E_OptionalUint32, uint32(32))
			proto.SetExtension(m, testpb.E_OptionalUint64, uint64(64))
			proto.SetExtension(m, testpb.E_OptionalFloat, float32(32.32))
			proto.SetExtension(m, testpb.E_OptionalDouble, float64(64.64))
			proto.SetExtension(m, testpb.E_OptionalBool, bool(true))
			proto.SetExtension(m, testpb.E_OptionalString, string("string"))
			proto.SetExtension(m, testpb.E_OptionalBytes, []byte("bytes"))
			proto.SetExtension(m, testpb.E_OptionalNestedMessage, &testpb.TestAllExtensions_NestedMessage{A: proto.Int32(-32)})
			proto.SetExtension(m, testpb.E_OptionalNestedEnum, testpb.TestAllTypes_NEG)
			return m
		}(),
		func() proto.Message {
			m := new(testpb.TestAllExtensions)
			proto.SetExtension(m, testpb.E_RepeatedInt32, []int32{-32, +32})
			proto.SetExtension(m, testpb.E_RepeatedInt64, []int64{-64, +64})
			proto.SetExtension(m, testpb.E_RepeatedUint32, []uint32{0, 32})
			proto.SetExtension(m, testpb.E_RepeatedUint64, []uint64{0, 64})
			proto.SetExtension(m, testpb.E_RepeatedFloat, []float32{-32.32, +32.32})
			proto.SetExtension(m, testpb.E_RepeatedDouble, []float64{-64.64, +64.64})
			proto.SetExtension(m, testpb.E_RepeatedBool, []bool{false, true})
			proto.SetExtension(m, testpb.E_RepeatedString, []string{"hello", "goodbye"})
			proto.SetExtension(m, testpb.E_RepeatedBytes, [][]byte{[]byte("hello"), []byte("goodbye")})
			proto.SetExtension(m, testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage{{A: proto.Int32(-32)}, {A: proto.Int32(+32)}})
			proto.SetExtension(m, testpb.E_RepeatedNestedEnum, []testpb.TestAllTypes_NestedEnum{testpb.TestAllTypes_FOO, testpb.TestAllTypes_NEG})
			return m
		}(),
		&textpb.KnownTypes{
			OptBool:   &wrapperspb.BoolValue{Value: true},
			OptInt32:  &wrapperspb.Int32Value{Value: -32},
			OptInt64:  &wrapperspb.Int64Value{Value: -64},
			OptUint32: &wrapperspb.UInt32Value{Value: +32},
			OptUint64: &wrapperspb.UInt64Value{Value: +64},
			OptFloat:  &wrapperspb.FloatValue{Value: 32.32},
			OptDouble: &wrapperspb.DoubleValue{Value: 64.64},
			OptString: &wrapperspb.StringValue{Value: "string"},
			OptBytes:  &wrapperspb.BytesValue{Value: []byte("bytes")},
		},
		&textpb.KnownTypes{
			OptAny: &anypb.Any{
				TypeUrl: "google.golang.org/goproto.proto.test.TestAllTypes",
				Value: func() []byte {
					b1, _ := proto.MarshalOptions{Deterministic: true}.Marshal(optMsg)
					b2, _ := proto.MarshalOptions{Deterministic: true}.Marshal(repMsg)
					b3, _ := proto.MarshalOptions{Deterministic: true}.Marshal(mapMsg)
					return append(append(append([]byte(nil), b1...), b2...), b3...)
				}(),
			},
		},
		&textpb.KnownTypes{
			OptAny: &anypb.Any{
				TypeUrl: "unknown_type",
				Value:   []byte("invalid_value"),
			},
		},
	}

	for _, src := range tests {
		dst := src.ProtoReflect().Type().New().Interface()
		proto.Merge(dst, newTransformer().transformMessage(src.ProtoReflect()))
		if diff := cmp.Diff(src, dst, Transform()); diff != "" {
			t.Errorf("Merge mismatch (-want +got):\n%s", diff)
		}
	}
}
