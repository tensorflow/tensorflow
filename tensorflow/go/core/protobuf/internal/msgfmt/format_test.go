// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package msgfmt_test

import (
	"math"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/msgfmt"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"
	"google.golang.org/protobuf/testing/protopack"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	textpb "google.golang.org/protobuf/internal/testprotos/textpb2"
	dynpb "google.golang.org/protobuf/types/dynamicpb"
	anypb "google.golang.org/protobuf/types/known/anypb"
	durpb "google.golang.org/protobuf/types/known/durationpb"
	tspb "google.golang.org/protobuf/types/known/timestamppb"
	wpb "google.golang.org/protobuf/types/known/wrapperspb"
)

func init() {
	detrand.Disable()
}

func TestFormat(t *testing.T) {
	optMsg := &testpb.TestAllTypes{
		OptionalBool:          proto.Bool(false),
		OptionalInt32:         proto.Int32(-32),
		OptionalInt64:         proto.Int64(-64),
		OptionalUint32:        proto.Uint32(32),
		OptionalUint64:        proto.Uint64(64),
		OptionalFloat:         proto.Float32(32.32),
		OptionalDouble:        proto.Float64(64.64),
		OptionalString:        proto.String("string"),
		OptionalBytes:         []byte("bytes"),
		OptionalNestedEnum:    testpb.TestAllTypes_NEG.Enum(),
		OptionalNestedMessage: &testpb.TestAllTypes_NestedMessage{A: proto.Int32(5)},
	}
	repMsg := &testpb.TestAllTypes{
		RepeatedBool:   []bool{false, true},
		RepeatedInt32:  []int32{32, -32},
		RepeatedInt64:  []int64{64, -64},
		RepeatedUint32: []uint32{0, 32},
		RepeatedUint64: []uint64{0, 64},
		RepeatedFloat:  []float32{0, 32.32},
		RepeatedDouble: []float64{0, 64.64},
		RepeatedString: []string{"s1", "s2"},
		RepeatedBytes:  [][]byte{{1}, {2}},
		RepeatedNestedEnum: []testpb.TestAllTypes_NestedEnum{
			testpb.TestAllTypes_FOO,
			testpb.TestAllTypes_BAR,
		},
		RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{
			{A: proto.Int32(5)},
			{A: proto.Int32(-5)},
		},
	}
	mapMsg := &testpb.TestAllTypes{
		MapBoolBool:     map[bool]bool{true: false},
		MapInt32Int32:   map[int32]int32{-32: 32},
		MapInt64Int64:   map[int64]int64{-64: 64},
		MapUint32Uint32: map[uint32]uint32{0: 32},
		MapUint64Uint64: map[uint64]uint64{0: 64},
		MapInt32Float:   map[int32]float32{32: 32.32},
		MapInt32Double:  map[int32]float64{64: 64.64},
		MapStringString: map[string]string{"k": "v"},
		MapStringBytes:  map[string][]byte{"k": []byte("v")},
		MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{
			"k": testpb.TestAllTypes_FOO,
		},
		MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{
			"k": {A: proto.Int32(5)},
		},
	}

	tests := []struct {
		in   proto.Message
		want string
	}{{
		in:   optMsg,
		want: `{optional_int32:-32, optional_int64:-64, optional_uint32:32, optional_uint64:64, optional_float:32.32, optional_double:64.64, optional_bool:false, optional_string:"string", optional_bytes:"bytes", optional_nested_message:{a:5}, optional_nested_enum:NEG}`,
	}, {
		in:   repMsg,
		want: `{repeated_int32:[32, -32], repeated_int64:[64, -64], repeated_uint32:[0, 32], repeated_uint64:[0, 64], repeated_float:[0, 32.32], repeated_double:[0, 64.64], repeated_bool:[false, true], repeated_string:["s1", "s2"], repeated_bytes:["\x01", "\x02"], repeated_nested_message:[{a:5}, {a:-5}], repeated_nested_enum:[FOO, BAR]}`,
	}, {
		in:   mapMsg,
		want: `{map_int32_int32:{-32:32}, map_int64_int64:{-64:64}, map_uint32_uint32:{0:32}, map_uint64_uint64:{0:64}, map_int32_float:{32:32.32}, map_int32_double:{64:64.64}, map_bool_bool:{true:false}, map_string_string:{"k":"v"}, map_string_bytes:{"k":"v"}, map_string_nested_message:{"k":{a:5}}, map_string_nested_enum:{"k":FOO}}`,
	}, {
		in: func() proto.Message {
			m := &testpb.TestAllExtensions{}
			proto.SetExtension(m, testpb.E_OptionalBool, bool(false))
			proto.SetExtension(m, testpb.E_OptionalInt32, int32(-32))
			proto.SetExtension(m, testpb.E_OptionalInt64, int64(-64))
			proto.SetExtension(m, testpb.E_OptionalUint32, uint32(32))
			proto.SetExtension(m, testpb.E_OptionalUint64, uint64(64))
			proto.SetExtension(m, testpb.E_OptionalFloat, float32(32.32))
			proto.SetExtension(m, testpb.E_OptionalDouble, float64(64.64))
			proto.SetExtension(m, testpb.E_OptionalString, string("string"))
			proto.SetExtension(m, testpb.E_OptionalBytes, []byte("bytes"))
			proto.SetExtension(m, testpb.E_OptionalNestedEnum, testpb.TestAllTypes_NEG)
			proto.SetExtension(m, testpb.E_OptionalNestedMessage, &testpb.TestAllExtensions_NestedMessage{A: proto.Int32(5)})
			return m
		}(),
		want: `{[goproto.proto.test.optional_bool]:false, [goproto.proto.test.optional_bytes]:"bytes", [goproto.proto.test.optional_double]:64.64, [goproto.proto.test.optional_float]:32.32, [goproto.proto.test.optional_int32]:-32, [goproto.proto.test.optional_int64]:-64, [goproto.proto.test.optional_nested_enum]:NEG, [goproto.proto.test.optional_nested_message]:{a:5}, [goproto.proto.test.optional_string]:"string", [goproto.proto.test.optional_uint32]:32, [goproto.proto.test.optional_uint64]:64}`,
	}, {
		in: func() proto.Message {
			m := &testpb.TestAllExtensions{}
			proto.SetExtension(m, testpb.E_RepeatedBool, []bool{false, true})
			proto.SetExtension(m, testpb.E_RepeatedInt32, []int32{32, -32})
			proto.SetExtension(m, testpb.E_RepeatedInt64, []int64{64, -64})
			proto.SetExtension(m, testpb.E_RepeatedUint32, []uint32{0, 32})
			proto.SetExtension(m, testpb.E_RepeatedUint64, []uint64{0, 64})
			proto.SetExtension(m, testpb.E_RepeatedFloat, []float32{0, 32.32})
			proto.SetExtension(m, testpb.E_RepeatedDouble, []float64{0, 64.64})
			proto.SetExtension(m, testpb.E_RepeatedString, []string{"s1", "s2"})
			proto.SetExtension(m, testpb.E_RepeatedBytes, [][]byte{{1}, {2}})
			proto.SetExtension(m, testpb.E_RepeatedNestedEnum, []testpb.TestAllTypes_NestedEnum{
				testpb.TestAllTypes_FOO,
				testpb.TestAllTypes_BAR,
			})
			proto.SetExtension(m, testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage{
				{A: proto.Int32(5)},
				{A: proto.Int32(-5)},
			})
			return m
		}(),
		want: `{[goproto.proto.test.repeated_bool]:[false, true], [goproto.proto.test.repeated_bytes]:["\x01", "\x02"], [goproto.proto.test.repeated_double]:[0, 64.64], [goproto.proto.test.repeated_float]:[0, 32.32], [goproto.proto.test.repeated_int32]:[32, -32], [goproto.proto.test.repeated_int64]:[64, -64], [goproto.proto.test.repeated_nested_enum]:[FOO, BAR], [goproto.proto.test.repeated_nested_message]:[{a:5}, {a:-5}], [goproto.proto.test.repeated_string]:["s1", "s2"], [goproto.proto.test.repeated_uint32]:[0, 32], [goproto.proto.test.repeated_uint64]:[0, 64]}`,
	}, {
		in: func() proto.Message {
			m := &testpb.TestAllTypes{}
			m.ProtoReflect().SetUnknown(protopack.Message{
				protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Uvarint(100),
				protopack.Tag{Number: 50001, Type: protopack.Fixed32Type}, protopack.Uint32(200),
				protopack.Tag{Number: 50002, Type: protopack.Fixed64Type}, protopack.Uint64(300),
				protopack.Tag{Number: 50003, Type: protopack.BytesType}, protopack.String("hello"),
				protopack.Message{
					protopack.Tag{Number: 50004, Type: protopack.StartGroupType},
					protopack.Tag{Number: 1, Type: protopack.VarintType}, protopack.Uvarint(100),
					protopack.Tag{Number: 1, Type: protopack.Fixed32Type}, protopack.Uint32(200),
					protopack.Tag{Number: 1, Type: protopack.Fixed64Type}, protopack.Uint64(300),
					protopack.Tag{Number: 1, Type: protopack.BytesType}, protopack.String("hello"),
					protopack.Message{
						protopack.Tag{Number: 1, Type: protopack.StartGroupType},
						protopack.Tag{Number: 1, Type: protopack.VarintType}, protopack.Uvarint(100),
						protopack.Tag{Number: 1, Type: protopack.Fixed32Type}, protopack.Uint32(200),
						protopack.Tag{Number: 1, Type: protopack.Fixed64Type}, protopack.Uint64(300),
						protopack.Tag{Number: 1, Type: protopack.BytesType}, protopack.String("hello"),
						protopack.Tag{Number: 1, Type: protopack.EndGroupType},
					},
					protopack.Tag{Number: 50004, Type: protopack.EndGroupType},
				},
			}.Marshal())
			return m
		}(),
		want: `{50000:100, 50001:0x000000c8, 50002:0x000000000000012c, 50003:"hello", 50004:{1:[100, 0x000000c8, 0x000000000000012c, "hello", {1:[100, 0x000000c8, 0x000000000000012c, "hello"]}]}}`,
	}, {
		in: &textpb.KnownTypes{
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
		want: `{opt_any:{[google.golang.org/goproto.proto.test.TestAllTypes]:{optional_int32:-32, optional_int64:-64, optional_uint32:32, optional_uint64:64, optional_float:32.32, optional_double:64.64, optional_bool:false, optional_string:"string", optional_bytes:"bytes", optional_nested_message:{a:5}, optional_nested_enum:NEG, repeated_int32:[32, -32], repeated_int64:[64, -64], repeated_uint32:[0, 32], repeated_uint64:[0, 64], repeated_float:[0, 32.32], repeated_double:[0, 64.64], repeated_bool:[false, true], repeated_string:["s1", "s2"], repeated_bytes:["\x01", "\x02"], repeated_nested_message:[{a:5}, {a:-5}], repeated_nested_enum:[FOO, BAR], map_int32_int32:{-32:32}, map_int64_int64:{-64:64}, map_uint32_uint32:{0:32}, map_uint64_uint64:{0:64}, map_int32_float:{32:32.32}, map_int32_double:{64:64.64}, map_bool_bool:{true:false}, map_string_string:{"k":"v"}, map_string_bytes:{"k":"v"}, map_string_nested_message:{"k":{a:5}}, map_string_nested_enum:{"k":FOO}}}}`,
	}, {
		in: &textpb.KnownTypes{
			OptTimestamp: &tspb.Timestamp{Seconds: math.MinInt64, Nanos: math.MaxInt32},
		},
		want: `{opt_timestamp:{seconds:-9223372036854775808, nanos:2147483647}}`,
	}, {
		in: &textpb.KnownTypes{
			OptTimestamp: &tspb.Timestamp{Seconds: 1257894123, Nanos: 456789},
		},
		want: `{opt_timestamp:2009-11-10T23:02:03.000456789Z}`,
	}, {
		in: &textpb.KnownTypes{
			OptDuration: &durpb.Duration{Seconds: math.MinInt64, Nanos: math.MaxInt32},
		},
		want: `{opt_duration:{seconds:-9223372036854775808, nanos:2147483647}}`,
	}, {
		in: &textpb.KnownTypes{
			OptDuration: &durpb.Duration{Seconds: +1257894123, Nanos: +456789},
		},
		want: `{opt_duration:1257894123.000456789s}`,
	}, {
		in: &textpb.KnownTypes{
			OptDuration: &durpb.Duration{Seconds: -1257894123, Nanos: -456789},
		},
		want: `{opt_duration:-1257894123.000456789s}`,
	}, {
		in: &textpb.KnownTypes{
			OptDuration: &durpb.Duration{Seconds: 0, Nanos: -1},
		},
		want: `{opt_duration:-0.000000001s}`,
	}, {
		in: &textpb.KnownTypes{
			OptBool:   &wpb.BoolValue{},
			OptInt32:  &wpb.Int32Value{},
			OptInt64:  &wpb.Int64Value{},
			OptUint32: &wpb.UInt32Value{},
			OptUint64: &wpb.UInt64Value{},
			OptFloat:  &wpb.FloatValue{},
			OptDouble: &wpb.DoubleValue{},
			OptString: &wpb.StringValue{},
			OptBytes:  &wpb.BytesValue{},
		},
		want: `{opt_bool:false, opt_int32:0, opt_int64:0, opt_uint32:0, opt_uint64:0, opt_float:0, opt_double:0, opt_string:"", opt_bytes:""}`,
	}}
	for _, tt := range tests {
		t.Run("Generated", func(t *testing.T) {
			got := msgfmt.Format(tt.in)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Format() mismatch (-want +got):\n%v", diff)
			}
		})
		t.Run("dynamicpb.Message", func(t *testing.T) {
			m := dynpb.NewMessage(tt.in.ProtoReflect().Descriptor())
			proto.Merge(m, tt.in)
			got := msgfmt.Format(m)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Format() mismatch (-want +got):\n%v", diff)
			}
		})
		t.Run("protocmp.Message", func(t *testing.T) {
			// This is a roundabout way to obtain a protocmp.Message since there
			// is no exported API in protocmp to directly transform a message.
			var m proto.Message
			var once sync.Once
			cmp.Equal(tt.in, tt.in, protocmp.Transform(), cmp.FilterPath(func(p cmp.Path) bool {
				if v, _ := p.Index(1).Values(); v.IsValid() {
					once.Do(func() { m = v.Interface().(protocmp.Message) })
				}
				return false
			}, cmp.Ignore()))

			got := msgfmt.Format(m)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Format() mismatch (-want +got):\n%v", diff)
			}
		})
	}
}
