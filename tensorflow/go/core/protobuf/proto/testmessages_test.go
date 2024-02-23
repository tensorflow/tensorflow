// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/internal/protobuild"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/testing/protopack"

	legacypb "google.golang.org/protobuf/internal/testprotos/legacy"
	requiredpb "google.golang.org/protobuf/internal/testprotos/required"
	testpb "google.golang.org/protobuf/internal/testprotos/test"
	test3pb "google.golang.org/protobuf/internal/testprotos/test3"
	testeditionspb "google.golang.org/protobuf/internal/testprotos/testeditions"
)

type testProto struct {
	desc             string
	decodeTo         []proto.Message
	wire             []byte
	partial          bool
	noEncode         bool
	checkFastInit    bool
	unmarshalOptions proto.UnmarshalOptions
	validationStatus impl.ValidationStatus
	nocheckValidInit bool
}

func makeMessages(in protobuild.Message, messages ...proto.Message) []proto.Message {
	if len(messages) == 0 {
		messages = []proto.Message{
			&testpb.TestAllTypes{},
			&test3pb.TestAllTypes{},
			&testpb.TestAllExtensions{},
			&testeditionspb.TestAllTypes{},
		}
	}
	for _, m := range messages {
		in.Build(m.ProtoReflect())
	}
	return messages
}

func templateMessages(messages ...proto.Message) []protoreflect.MessageType {
	if len(messages) == 0 {
		messages = []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*test3pb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
			(*testeditionspb.TestAllTypes)(nil),
		}
	}
	var out []protoreflect.MessageType
	for _, m := range messages {
		out = append(out, m.ProtoReflect().Type())
	}
	return out

}

var testValidMessages = []testProto{
	{
		desc:          "basic scalar types",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"optional_int32":       1001,
			"optional_int64":       1002,
			"optional_uint32":      1003,
			"optional_uint64":      1004,
			"optional_sint32":      1005,
			"optional_sint64":      1006,
			"optional_fixed32":     1007,
			"optional_fixed64":     1008,
			"optional_sfixed32":    1009,
			"optional_sfixed64":    1010,
			"optional_float":       1011.5,
			"optional_double":      1012.5,
			"optional_bool":        true,
			"optional_string":      "string",
			"optional_bytes":       []byte("bytes"),
			"optional_nested_enum": "BAR",
		}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(1001),
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1002),
			protopack.Tag{3, protopack.VarintType}, protopack.Uvarint(1003),
			protopack.Tag{4, protopack.VarintType}, protopack.Uvarint(1004),
			protopack.Tag{5, protopack.VarintType}, protopack.Svarint(1005),
			protopack.Tag{6, protopack.VarintType}, protopack.Svarint(1006),
			protopack.Tag{7, protopack.Fixed32Type}, protopack.Uint32(1007),
			protopack.Tag{8, protopack.Fixed64Type}, protopack.Uint64(1008),
			protopack.Tag{9, protopack.Fixed32Type}, protopack.Int32(1009),
			protopack.Tag{10, protopack.Fixed64Type}, protopack.Int64(1010),
			protopack.Tag{11, protopack.Fixed32Type}, protopack.Float32(1011.5),
			protopack.Tag{12, protopack.Fixed64Type}, protopack.Float64(1012.5),
			protopack.Tag{13, protopack.VarintType}, protopack.Bool(true),
			protopack.Tag{14, protopack.BytesType}, protopack.String("string"),
			protopack.Tag{15, protopack.BytesType}, protopack.Bytes([]byte("bytes")),
			protopack.Tag{21, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_BAR)),
		}.Marshal(),
	},
	{
		desc: "zero values",
		decodeTo: makeMessages(protobuild.Message{
			"optional_int32":    0,
			"optional_int64":    0,
			"optional_uint32":   0,
			"optional_uint64":   0,
			"optional_sint32":   0,
			"optional_sint64":   0,
			"optional_fixed32":  0,
			"optional_fixed64":  0,
			"optional_sfixed32": 0,
			"optional_sfixed64": 0,
			"optional_float":    0,
			"optional_double":   0,
			"optional_bool":     false,
			"optional_string":   "",
			"optional_bytes":    []byte{},
		}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(0),
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(0),
			protopack.Tag{3, protopack.VarintType}, protopack.Uvarint(0),
			protopack.Tag{4, protopack.VarintType}, protopack.Uvarint(0),
			protopack.Tag{5, protopack.VarintType}, protopack.Svarint(0),
			protopack.Tag{6, protopack.VarintType}, protopack.Svarint(0),
			protopack.Tag{7, protopack.Fixed32Type}, protopack.Uint32(0),
			protopack.Tag{8, protopack.Fixed64Type}, protopack.Uint64(0),
			protopack.Tag{9, protopack.Fixed32Type}, protopack.Int32(0),
			protopack.Tag{10, protopack.Fixed64Type}, protopack.Int64(0),
			protopack.Tag{11, protopack.Fixed32Type}, protopack.Float32(0),
			protopack.Tag{12, protopack.Fixed64Type}, protopack.Float64(0),
			protopack.Tag{13, protopack.VarintType}, protopack.Bool(false),
			protopack.Tag{14, protopack.BytesType}, protopack.String(""),
			protopack.Tag{15, protopack.BytesType}, protopack.Bytes(nil),
		}.Marshal(),
	},
	{
		desc: "proto3 zero values",
		decodeTo: makeMessages(protobuild.Message{
			"singular_int32":    0,
			"singular_int64":    0,
			"singular_uint32":   0,
			"singular_uint64":   0,
			"singular_sint32":   0,
			"singular_sint64":   0,
			"singular_fixed32":  0,
			"singular_fixed64":  0,
			"singular_sfixed32": 0,
			"singular_sfixed64": 0,
			"singular_float":    0,
			"singular_double":   0,
			"singular_bool":     false,
			"singular_string":   "",
			"singular_bytes":    []byte{},
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{81, protopack.VarintType}, protopack.Varint(0),
			protopack.Tag{82, protopack.VarintType}, protopack.Varint(0),
			protopack.Tag{83, protopack.VarintType}, protopack.Uvarint(0),
			protopack.Tag{84, protopack.VarintType}, protopack.Uvarint(0),
			protopack.Tag{85, protopack.VarintType}, protopack.Svarint(0),
			protopack.Tag{86, protopack.VarintType}, protopack.Svarint(0),
			protopack.Tag{87, protopack.Fixed32Type}, protopack.Uint32(0),
			protopack.Tag{88, protopack.Fixed64Type}, protopack.Uint64(0),
			protopack.Tag{89, protopack.Fixed32Type}, protopack.Int32(0),
			protopack.Tag{90, protopack.Fixed64Type}, protopack.Int64(0),
			protopack.Tag{91, protopack.Fixed32Type}, protopack.Float32(0),
			protopack.Tag{92, protopack.Fixed64Type}, protopack.Float64(0),
			protopack.Tag{93, protopack.VarintType}, protopack.Bool(false),
			protopack.Tag{94, protopack.BytesType}, protopack.String(""),
			protopack.Tag{95, protopack.BytesType}, protopack.Bytes(nil),
		}.Marshal(),
	},
	{
		desc: "groups",
		decodeTo: makeMessages(protobuild.Message{
			"optionalgroup": protobuild.Message{
				"a":                 1017,
				"same_field_number": 1016,
			},
		}, &testpb.TestAllTypes{}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{16, protopack.StartGroupType},
			protopack.Tag{17, protopack.VarintType}, protopack.Varint(1017),
			protopack.Tag{16, protopack.VarintType}, protopack.Varint(1016),
			protopack.Tag{16, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "groups (field overridden)",
		decodeTo: makeMessages(protobuild.Message{
			"optionalgroup": protobuild.Message{
				"a": 2,
			},
		}, &testpb.TestAllTypes{}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{16, protopack.StartGroupType},
			protopack.Tag{17, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{16, protopack.EndGroupType},
			protopack.Tag{16, protopack.StartGroupType},
			protopack.Tag{17, protopack.VarintType}, protopack.Varint(2),
			protopack.Tag{16, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "messages",
		decodeTo: makeMessages(protobuild.Message{
			"optional_nested_message": protobuild.Message{
				"a": 42,
				"corecursive": protobuild.Message{
					"optional_int32": 43,
				},
			},
		}),
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(42),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(43),
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "messages (split across multiple tags)",
		decodeTo: makeMessages(protobuild.Message{
			"optional_nested_message": protobuild.Message{
				"a": 42,
				"corecursive": protobuild.Message{
					"optional_int32": 43,
				},
			},
		}),
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(42),
			}),
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(43),
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "messages (field overridden)",
		decodeTo: makeMessages(protobuild.Message{
			"optional_nested_message": protobuild.Message{
				"a": 2,
			},
		}),
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc: "basic repeated types",
		decodeTo: makeMessages(protobuild.Message{
			"repeated_int32":       []int32{1001, 2001},
			"repeated_int64":       []int64{1002, 2002},
			"repeated_uint32":      []uint32{1003, 2003},
			"repeated_uint64":      []uint64{1004, 2004},
			"repeated_sint32":      []int32{1005, 2005},
			"repeated_sint64":      []int64{1006, 2006},
			"repeated_fixed32":     []uint32{1007, 2007},
			"repeated_fixed64":     []uint64{1008, 2008},
			"repeated_sfixed32":    []int32{1009, 2009},
			"repeated_sfixed64":    []int64{1010, 2010},
			"repeated_float":       []float32{1011.5, 2011.5},
			"repeated_double":      []float64{1012.5, 2012.5},
			"repeated_bool":        []bool{true, false},
			"repeated_string":      []string{"foo", "bar"},
			"repeated_bytes":       []string{"FOO", "BAR"},
			"repeated_nested_enum": []string{"FOO", "BAR"},
		}),
		wire: protopack.Message{
			protopack.Tag{31, protopack.VarintType}, protopack.Varint(1001),
			protopack.Tag{31, protopack.VarintType}, protopack.Varint(2001),
			protopack.Tag{32, protopack.VarintType}, protopack.Varint(1002),
			protopack.Tag{32, protopack.VarintType}, protopack.Varint(2002),
			protopack.Tag{33, protopack.VarintType}, protopack.Uvarint(1003),
			protopack.Tag{33, protopack.VarintType}, protopack.Uvarint(2003),
			protopack.Tag{34, protopack.VarintType}, protopack.Uvarint(1004),
			protopack.Tag{34, protopack.VarintType}, protopack.Uvarint(2004),
			protopack.Tag{35, protopack.VarintType}, protopack.Svarint(1005),
			protopack.Tag{35, protopack.VarintType}, protopack.Svarint(2005),
			protopack.Tag{36, protopack.VarintType}, protopack.Svarint(1006),
			protopack.Tag{36, protopack.VarintType}, protopack.Svarint(2006),
			protopack.Tag{37, protopack.Fixed32Type}, protopack.Uint32(1007),
			protopack.Tag{37, protopack.Fixed32Type}, protopack.Uint32(2007),
			protopack.Tag{38, protopack.Fixed64Type}, protopack.Uint64(1008),
			protopack.Tag{38, protopack.Fixed64Type}, protopack.Uint64(2008),
			protopack.Tag{39, protopack.Fixed32Type}, protopack.Int32(1009),
			protopack.Tag{39, protopack.Fixed32Type}, protopack.Int32(2009),
			protopack.Tag{40, protopack.Fixed64Type}, protopack.Int64(1010),
			protopack.Tag{40, protopack.Fixed64Type}, protopack.Int64(2010),
			protopack.Tag{41, protopack.Fixed32Type}, protopack.Float32(1011.5),
			protopack.Tag{41, protopack.Fixed32Type}, protopack.Float32(2011.5),
			protopack.Tag{42, protopack.Fixed64Type}, protopack.Float64(1012.5),
			protopack.Tag{42, protopack.Fixed64Type}, protopack.Float64(2012.5),
			protopack.Tag{43, protopack.VarintType}, protopack.Bool(true),
			protopack.Tag{43, protopack.VarintType}, protopack.Bool(false),
			protopack.Tag{44, protopack.BytesType}, protopack.String("foo"),
			protopack.Tag{44, protopack.BytesType}, protopack.String("bar"),
			protopack.Tag{45, protopack.BytesType}, protopack.Bytes([]byte("FOO")),
			protopack.Tag{45, protopack.BytesType}, protopack.Bytes([]byte("BAR")),
			protopack.Tag{51, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_FOO)),
			protopack.Tag{51, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_BAR)),
		}.Marshal(),
	},
	{
		desc: "basic repeated types (packed encoding)",
		decodeTo: makeMessages(protobuild.Message{
			"repeated_int32":       []int32{1001, 2001},
			"repeated_int64":       []int64{1002, 2002},
			"repeated_uint32":      []uint32{1003, 2003},
			"repeated_uint64":      []uint64{1004, 2004},
			"repeated_sint32":      []int32{1005, 2005},
			"repeated_sint64":      []int64{1006, 2006},
			"repeated_fixed32":     []uint32{1007, 2007},
			"repeated_fixed64":     []uint64{1008, 2008},
			"repeated_sfixed32":    []int32{1009, 2009},
			"repeated_sfixed64":    []int64{1010, 2010},
			"repeated_float":       []float32{1011.5, 2011.5},
			"repeated_double":      []float64{1012.5, 2012.5},
			"repeated_bool":        []bool{true, false},
			"repeated_nested_enum": []string{"FOO", "BAR"},
		}),
		wire: protopack.Message{
			protopack.Tag{31, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(1001), protopack.Varint(2001),
			},
			protopack.Tag{32, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(1002), protopack.Varint(2002),
			},
			protopack.Tag{33, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uvarint(1003), protopack.Uvarint(2003),
			},
			protopack.Tag{34, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uvarint(1004), protopack.Uvarint(2004),
			},
			protopack.Tag{35, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Svarint(1005), protopack.Svarint(2005),
			},
			protopack.Tag{36, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Svarint(1006), protopack.Svarint(2006),
			},
			protopack.Tag{37, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uint32(1007), protopack.Uint32(2007),
			},
			protopack.Tag{38, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uint64(1008), protopack.Uint64(2008),
			},
			protopack.Tag{39, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Int32(1009), protopack.Int32(2009),
			},
			protopack.Tag{40, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Int64(1010), protopack.Int64(2010),
			},
			protopack.Tag{41, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Float32(1011.5), protopack.Float32(2011.5),
			},
			protopack.Tag{42, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Float64(1012.5), protopack.Float64(2012.5),
			},
			protopack.Tag{43, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Bool(true), protopack.Bool(false),
			},
			protopack.Tag{51, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(int(testpb.TestAllTypes_FOO)),
				protopack.Varint(int(testpb.TestAllTypes_BAR)),
			},
		}.Marshal(),
	},
	{
		desc: "basic repeated types (zero-length packed encoding)",
		decodeTo: makeMessages(protobuild.Message{
			"repeated_int32":       []int32{},
			"repeated_int64":       []int64{},
			"repeated_uint32":      []uint32{},
			"repeated_uint64":      []uint64{},
			"repeated_sint32":      []int32{},
			"repeated_sint64":      []int64{},
			"repeated_fixed32":     []uint32{},
			"repeated_fixed64":     []uint64{},
			"repeated_sfixed32":    []int32{},
			"repeated_sfixed64":    []int64{},
			"repeated_float":       []float32{},
			"repeated_double":      []float64{},
			"repeated_bool":        []bool{},
			"repeated_nested_enum": []string{},
		}),
		wire: protopack.Message{
			protopack.Tag{31, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{32, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{33, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{34, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{35, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{36, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{37, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{38, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{39, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{40, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{41, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{42, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{43, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{51, protopack.BytesType}, protopack.LengthPrefix{},
		}.Marshal(),
	},
	{
		desc: "packed repeated types",
		decodeTo: makeMessages(protobuild.Message{
			"packed_int32":    []int32{1001, 2001},
			"packed_int64":    []int64{1002, 2002},
			"packed_uint32":   []uint32{1003, 2003},
			"packed_uint64":   []uint64{1004, 2004},
			"packed_sint32":   []int32{1005, 2005},
			"packed_sint64":   []int64{1006, 2006},
			"packed_fixed32":  []uint32{1007, 2007},
			"packed_fixed64":  []uint64{1008, 2008},
			"packed_sfixed32": []int32{1009, 2009},
			"packed_sfixed64": []int64{1010, 2010},
			"packed_float":    []float32{1011.5, 2011.5},
			"packed_double":   []float64{1012.5, 2012.5},
			"packed_bool":     []bool{true, false},
			"packed_enum":     []string{"FOREIGN_FOO", "FOREIGN_BAR"},
		}, &testpb.TestPackedTypes{}, &testpb.TestPackedExtensions{}),
		wire: protopack.Message{
			protopack.Tag{90, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(1001), protopack.Varint(2001),
			},
			protopack.Tag{91, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(1002), protopack.Varint(2002),
			},
			protopack.Tag{92, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uvarint(1003), protopack.Uvarint(2003),
			},
			protopack.Tag{93, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uvarint(1004), protopack.Uvarint(2004),
			},
			protopack.Tag{94, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Svarint(1005), protopack.Svarint(2005),
			},
			protopack.Tag{95, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Svarint(1006), protopack.Svarint(2006),
			},
			protopack.Tag{96, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uint32(1007), protopack.Uint32(2007),
			},
			protopack.Tag{97, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Uint64(1008), protopack.Uint64(2008),
			},
			protopack.Tag{98, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Int32(1009), protopack.Int32(2009),
			},
			protopack.Tag{99, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Int64(1010), protopack.Int64(2010),
			},
			protopack.Tag{100, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Float32(1011.5), protopack.Float32(2011.5),
			},
			protopack.Tag{101, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Float64(1012.5), protopack.Float64(2012.5),
			},
			protopack.Tag{102, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Bool(true), protopack.Bool(false),
			},
			protopack.Tag{103, protopack.BytesType}, protopack.LengthPrefix{
				protopack.Varint(int(testpb.ForeignEnum_FOREIGN_FOO)),
				protopack.Varint(int(testpb.ForeignEnum_FOREIGN_BAR)),
			},
		}.Marshal(),
	},
	{
		desc: "packed repeated types (zero length)",
		decodeTo: makeMessages(protobuild.Message{
			"packed_int32":    []int32{},
			"packed_int64":    []int64{},
			"packed_uint32":   []uint32{},
			"packed_uint64":   []uint64{},
			"packed_sint32":   []int32{},
			"packed_sint64":   []int64{},
			"packed_fixed32":  []uint32{},
			"packed_fixed64":  []uint64{},
			"packed_sfixed32": []int32{},
			"packed_sfixed64": []int64{},
			"packed_float":    []float32{},
			"packed_double":   []float64{},
			"packed_bool":     []bool{},
			"packed_enum":     []string{},
		}, &testpb.TestPackedTypes{}, &testpb.TestPackedExtensions{}),
		wire: protopack.Message{
			protopack.Tag{90, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{91, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{92, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{93, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{94, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{95, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{96, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{97, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{98, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{99, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{100, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{101, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{102, protopack.BytesType}, protopack.LengthPrefix{},
			protopack.Tag{103, protopack.BytesType}, protopack.LengthPrefix{},
		}.Marshal(),
	},
	{
		desc: "repeated messages",
		decodeTo: makeMessages(protobuild.Message{
			"repeated_nested_message": []protobuild.Message{
				{"a": 1},
				{},
				{"a": 2},
			},
		}),
		wire: protopack.Message{
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc: "repeated nil messages",
		decodeTo: []proto.Message{&testpb.TestAllTypes{
			RepeatedNestedMessage: []*testpb.TestAllTypes_NestedMessage{
				{A: proto.Int32(1)},
				nil,
				{A: proto.Int32(2)},
			},
		}, &test3pb.TestAllTypes{
			RepeatedNestedMessage: []*test3pb.TestAllTypes_NestedMessage{
				{A: 1},
				nil,
				{A: 2},
			},
		}, build(
			&testpb.TestAllExtensions{},
			extend(testpb.E_RepeatedNestedMessage, []*testpb.TestAllExtensions_NestedMessage{
				{A: proto.Int32(1)},
				nil,
				{A: proto.Int32(2)},
			}),
		)},
		wire: protopack.Message{
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc: "repeated groups",
		decodeTo: makeMessages(protobuild.Message{
			"repeatedgroup": []protobuild.Message{
				{"a": 1017},
				{},
				{"a": 2017},
			},
		}, &testpb.TestAllTypes{}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{47, protopack.VarintType}, protopack.Varint(1017),
			protopack.Tag{46, protopack.EndGroupType},
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{46, protopack.EndGroupType},
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{47, protopack.VarintType}, protopack.Varint(2017),
			protopack.Tag{46, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "repeated nil groups",
		decodeTo: []proto.Message{&testpb.TestAllTypes{
			Repeatedgroup: []*testpb.TestAllTypes_RepeatedGroup{
				{A: proto.Int32(1017)},
				nil,
				{A: proto.Int32(2017)},
			},
		}, build(
			&testpb.TestAllExtensions{},
			extend(testpb.E_Repeatedgroup, []*testpb.RepeatedGroup{
				{A: proto.Int32(1017)},
				nil,
				{A: proto.Int32(2017)},
			}),
		)},
		wire: protopack.Message{
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{47, protopack.VarintType}, protopack.Varint(1017),
			protopack.Tag{46, protopack.EndGroupType},
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{46, protopack.EndGroupType},
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{47, protopack.VarintType}, protopack.Varint(2017),
			protopack.Tag{46, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "maps",
		decodeTo: makeMessages(protobuild.Message{
			"map_int32_int32":       map[int32]int32{1056: 1156, 2056: 2156},
			"map_int64_int64":       map[int64]int64{1057: 1157, 2057: 2157},
			"map_uint32_uint32":     map[uint32]uint32{1058: 1158, 2058: 2158},
			"map_uint64_uint64":     map[uint64]uint64{1059: 1159, 2059: 2159},
			"map_sint32_sint32":     map[int32]int32{1060: 1160, 2060: 2160},
			"map_sint64_sint64":     map[int64]int64{1061: 1161, 2061: 2161},
			"map_fixed32_fixed32":   map[uint32]uint32{1062: 1162, 2062: 2162},
			"map_fixed64_fixed64":   map[uint64]uint64{1063: 1163, 2063: 2163},
			"map_sfixed32_sfixed32": map[int32]int32{1064: 1164, 2064: 2164},
			"map_sfixed64_sfixed64": map[int64]int64{1065: 1165, 2065: 2165},
			"map_int32_float":       map[int32]float32{1066: 1166.5, 2066: 2166.5},
			"map_int32_double":      map[int32]float64{1067: 1167.5, 2067: 2167.5},
			"map_bool_bool":         map[bool]bool{true: false, false: true},
			"map_string_string":     map[string]string{"69.1.key": "69.1.val", "69.2.key": "69.2.val"},
			"map_string_bytes":      map[string][]byte{"70.1.key": []byte("70.1.val"), "70.2.key": []byte("70.2.val")},
			"map_string_nested_message": map[string]protobuild.Message{
				"71.1.key": {"a": 1171},
				"71.2.key": {"a": 2171},
			},
			"map_string_nested_enum": map[string]string{"73.1.key": "FOO", "73.2.key": "BAR"},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1056),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1156),
			}),
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2056),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(2156),
			}),
			protopack.Tag{57, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1057),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1157),
			}),
			protopack.Tag{57, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2057),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(2157),
			}),
			protopack.Tag{58, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1058),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1158),
			}),
			protopack.Tag{58, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2058),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(2158),
			}),
			protopack.Tag{59, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1059),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1159),
			}),
			protopack.Tag{59, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2059),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(2159),
			}),
			protopack.Tag{60, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Svarint(1060),
				protopack.Tag{2, protopack.VarintType}, protopack.Svarint(1160),
			}),
			protopack.Tag{60, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Svarint(2060),
				protopack.Tag{2, protopack.VarintType}, protopack.Svarint(2160),
			}),
			protopack.Tag{61, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Svarint(1061),
				protopack.Tag{2, protopack.VarintType}, protopack.Svarint(1161),
			}),
			protopack.Tag{61, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Svarint(2061),
				protopack.Tag{2, protopack.VarintType}, protopack.Svarint(2161),
			}),
			protopack.Tag{62, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(1062),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Int32(1162),
			}),
			protopack.Tag{62, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(2062),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Int32(2162),
			}),
			protopack.Tag{63, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed64Type}, protopack.Int64(1063),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Int64(1163),
			}),
			protopack.Tag{63, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed64Type}, protopack.Int64(2063),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Int64(2163),
			}),
			protopack.Tag{64, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(1064),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Int32(1164),
			}),
			protopack.Tag{64, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(2064),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Int32(2164),
			}),
			protopack.Tag{65, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed64Type}, protopack.Int64(1065),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Int64(1165),
			}),
			protopack.Tag{65, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.Fixed64Type}, protopack.Int64(2065),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Int64(2165),
			}),
			protopack.Tag{66, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1066),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Float32(1166.5),
			}),
			protopack.Tag{66, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2066),
				protopack.Tag{2, protopack.Fixed32Type}, protopack.Float32(2166.5),
			}),
			protopack.Tag{67, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1067),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Float64(1167.5),
			}),
			protopack.Tag{67, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2067),
				protopack.Tag{2, protopack.Fixed64Type}, protopack.Float64(2167.5),
			}),
			protopack.Tag{68, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Bool(true),
				protopack.Tag{2, protopack.VarintType}, protopack.Bool(false),
			}),
			protopack.Tag{68, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Bool(false),
				protopack.Tag{2, protopack.VarintType}, protopack.Bool(true),
			}),
			protopack.Tag{69, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("69.1.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("69.1.val"),
			}),
			protopack.Tag{69, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("69.2.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("69.2.val"),
			}),
			protopack.Tag{70, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("70.1.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("70.1.val"),
			}),
			protopack.Tag{70, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("70.2.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("70.2.val"),
			}),
			protopack.Tag{71, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("71.1.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1171),
				}),
			}),
			protopack.Tag{71, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("71.2.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(2171),
				}),
			}),
			protopack.Tag{73, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("73.1.key"),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_FOO)),
			}),
			protopack.Tag{73, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("73.2.key"),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_BAR)),
			}),
		}.Marshal(),
	},
	{
		desc: "map with value before key",
		decodeTo: makeMessages(protobuild.Message{
			"map_int32_int32": map[int32]int32{1056: 1156},
			"map_string_nested_message": map[string]protobuild.Message{
				"71.1.key": {"a": 1171},
			},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1156),
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1056),
			}),
			protopack.Tag{71, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1171),
				}),
				protopack.Tag{1, protopack.BytesType}, protopack.String("71.1.key"),
			}),
		}.Marshal(),
	},
	{
		desc: "map with repeated key and value",
		decodeTo: makeMessages(protobuild.Message{
			"map_int32_int32": map[int32]int32{1056: 1156},
			"map_string_nested_message": map[string]protobuild.Message{
				"71.1.key": {"a": 1171},
			},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1056),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1156),
			}),
			protopack.Tag{71, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String(""),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.BytesType}, protopack.String("71.1.key"),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1171),
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "oneof (uint32)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_uint32": 1111,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{111, protopack.VarintType}, protopack.Varint(1111)}.Marshal(),
	},
	{
		desc: "oneof (message)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_nested_message": protobuild.Message{
				"a": 1112,
			},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{112, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
			protopack.Message{protopack.Tag{1, protopack.VarintType}, protopack.Varint(1112)},
		})}.Marshal(),
	},
	{
		desc: "oneof (empty message)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_nested_message": protobuild.Message{},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{112, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{})}.Marshal(),
	},
	{
		desc: "oneof (merged message)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_nested_message": protobuild.Message{
				"a": 1,
				"corecursive": protobuild.Message{
					"optional_int32": 43,
				},
			},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{112, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Message{protopack.Tag{1, protopack.VarintType}, protopack.Varint(1)},
			}),
			protopack.Tag{112, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(43),
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "oneof (group)",
		decodeTo: makeMessages(protobuild.Message{
			"oneofgroup": protobuild.Message{
				"a": 1,
			},
		}, &testpb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{121, protopack.StartGroupType},
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{121, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "oneof (empty group)",
		decodeTo: makeMessages(protobuild.Message{
			"oneofgroup": protobuild.Message{},
		}, &testpb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{121, protopack.StartGroupType},
			protopack.Tag{121, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "oneof (merged group)",
		decodeTo: makeMessages(protobuild.Message{
			"oneofgroup": protobuild.Message{
				"a": 1,
				"b": 2,
			},
		}, &testpb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{121, protopack.StartGroupType},
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{121, protopack.EndGroupType},
			protopack.Tag{121, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(2),
			protopack.Tag{121, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "oneof (string)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_string": "1113",
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{113, protopack.BytesType}, protopack.String("1113")}.Marshal(),
	},
	{
		desc: "oneof (bytes)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_bytes": "1114",
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{114, protopack.BytesType}, protopack.String("1114")}.Marshal(),
	},
	{
		desc: "oneof (bool)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_bool": true,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{115, protopack.VarintType}, protopack.Bool(true)}.Marshal(),
	},
	{
		desc: "oneof (uint64)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_uint64": 116,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{116, protopack.VarintType}, protopack.Varint(116)}.Marshal(),
	},
	{
		desc: "oneof (float)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_float": 117.5,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{117, protopack.Fixed32Type}, protopack.Float32(117.5)}.Marshal(),
	},
	{
		desc: "oneof (double)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_double": 118.5,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{118, protopack.Fixed64Type}, protopack.Float64(118.5)}.Marshal(),
	},
	{
		desc: "oneof (enum)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_enum": "BAR",
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{119, protopack.VarintType}, protopack.Varint(int(testpb.TestAllTypes_BAR))}.Marshal(),
	},
	{
		desc: "oneof (zero)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_uint64": 0,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{116, protopack.VarintType}, protopack.Varint(0)}.Marshal(),
	},
	{
		desc: "oneof (overridden value)",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_uint64": 2,
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{111, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{116, protopack.VarintType}, protopack.Varint(2),
		}.Marshal(),
	},
	// TODO: More unknown field tests for ordering, repeated fields, etc.
	//
	// It is currently impossible to produce results that the v1 Equal
	// considers equivalent to those of the v1 decoder. Figure out if
	// that's a problem or not.
	{
		desc:          "unknown fields",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			protobuild.Unknown: protopack.Message{
				protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
			}.Marshal(),
		}),
		wire: protopack.Message{
			protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
		}.Marshal(),
	},
	{
		desc: "discarded unknown fields",
		unmarshalOptions: proto.UnmarshalOptions{
			DiscardUnknown: true,
		},
		decodeTo: makeMessages(protobuild.Message{}),
		wire: protopack.Message{
			protopack.Tag{100000, protopack.VarintType}, protopack.Varint(1),
		}.Marshal(),
	},
	{
		desc: "field type mismatch",
		decodeTo: makeMessages(protobuild.Message{
			protobuild.Unknown: protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("string"),
			}.Marshal(),
		}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.String("string"),
		}.Marshal(),
	},
	{
		desc: "map field element mismatch",
		decodeTo: makeMessages(protobuild.Message{
			"map_int32_int32": map[int32]int32{1: 0},
		}, &testpb.TestAllTypes{}, &test3pb.TestAllTypes{}, &testeditionspb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				protopack.Tag{2, protopack.BytesType}, protopack.String("string"),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in nil message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      []proto.Message{(*testpb.TestRequired)(nil)},
	},
	{
		desc:          "required int32 unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Int32{}),
	},
	{
		desc:          "required int32 set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"v": 1,
		}, &requiredpb.Int32{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
		}.Marshal(),
	},
	{
		desc:          "required fixed32 unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Fixed32{}),
	},
	{
		desc:          "required fixed32 set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"v": 1,
		}, &requiredpb.Fixed32{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(1),
		}.Marshal(),
	},
	{
		desc:          "required fixed64 unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Fixed64{}),
	},
	{
		desc:          "required fixed64 set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"v": 1,
		}, &requiredpb.Fixed64{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.Fixed64Type}, protopack.Int64(1),
		}.Marshal(),
	},
	{
		desc:          "required bytes unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Bytes{}),
	},
	{
		desc:          "required bytes set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"v": "",
		}, &requiredpb.Bytes{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.Bytes(nil),
		}.Marshal(),
	},
	{
		desc:          "required message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Message{}),
	},
	{
		desc:          "required message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"v": protobuild.Message{},
		}, &requiredpb.Message{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
		}.Marshal(),
	},
	{
		desc:          "required group unset",
		checkFastInit: true,
		partial:       true,
		decodeTo:      makeMessages(protobuild.Message{}, &requiredpb.Group{}),
	},
	{
		desc:          "required group set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"group": protobuild.Message{},
		}, &requiredpb.Group{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc:          "required field with incompatible wire type",
		checkFastInit: true,
		partial:       true,
		decodeTo: []proto.Message{build(
			&testpb.TestRequired{},
			unknown(protopack.Message{
				protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(2),
			}.Marshal()),
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.Fixed32Type}, protopack.Int32(2),
		}.Marshal(),
	},
	{
		desc:          "required field in optional message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"optional_message": protobuild.Message{},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
		}.Marshal(),
	},
	{
		desc:          "required field in optional message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"optional_message": protobuild.Message{
				"required_field": 1,
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
		}.Marshal(),
	},
	{
		desc:             "required field in optional message set (split across multiple tags)",
		checkFastInit:    false, // fast init checks don't handle split messages
		nocheckValidInit: true,  // validation doesn't either
		decodeTo: makeMessages(protobuild.Message{
			"optional_message": protobuild.Message{
				"required_field": 1,
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in repeated message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"repeated_message": []protobuild.Message{
				{"required_field": 1},
				{},
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
		}.Marshal(),
	},
	{
		desc:          "required field in repeated message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"repeated_message": []protobuild.Message{
				{"required_field": 1},
				{"required_field": 2},
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in map message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"map_message": map[int32]protobuild.Message{
				1: {"required_field": 1},
				2: {},
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				}),
			}),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in absent map message value",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"map_message": map[int32]protobuild.Message{
				2: {},
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in map message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"map_message": map[int32]protobuild.Message{
				1: {"required_field": 1},
				2: {"required_field": 2},
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				}),
			}),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
				}),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in optional group unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"optionalgroup": protobuild.Message{},
		}, &testpb.TestRequiredGroupFields{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc:          "required field in optional group set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"optionalgroup": protobuild.Message{
				"a": 1,
			},
		}, &testpb.TestRequiredGroupFields{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc:          "required field in repeated group unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"repeatedgroup": []protobuild.Message{
				{"a": 1},
				{},
			},
		}, &testpb.TestRequiredGroupFields{}),
		wire: protopack.Message{
			protopack.Tag{3, protopack.StartGroupType},
			protopack.Tag{4, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{3, protopack.EndGroupType},
			protopack.Tag{3, protopack.StartGroupType},
			protopack.Tag{3, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc:          "required field in repeated group set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"repeatedgroup": []protobuild.Message{
				{"a": 1},
				{"a": 2},
			},
		}, &testpb.TestRequiredGroupFields{}),
		wire: protopack.Message{
			protopack.Tag{3, protopack.StartGroupType},
			protopack.Tag{4, protopack.VarintType}, protopack.Varint(1),
			protopack.Tag{3, protopack.EndGroupType},
			protopack.Tag{3, protopack.StartGroupType},
			protopack.Tag{4, protopack.VarintType}, protopack.Varint(2),
			protopack.Tag{3, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc:          "required field in oneof message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"oneof_message": protobuild.Message{},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{protopack.Tag{4, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{})}.Marshal(),
	},
	{
		desc:          "required field in oneof message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"oneof_message": protobuild.Message{
				"required_field": 1,
			},
		}, &testpb.TestRequiredForeign{}),
		wire: protopack.Message{protopack.Tag{4, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
			protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
		})}.Marshal(),
	},
	{
		desc:          "required field in extension message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"single": protobuild.Message{},
		}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{1000, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
		}.Marshal(),
	},
	{
		desc:          "required field in extension message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"single": protobuild.Message{
				"required_field": 1,
			},
		}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{1000, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
		}.Marshal(),
	},
	{
		desc:          "required field in repeated extension message unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: makeMessages(protobuild.Message{
			"multi": []protobuild.Message{
				{"required_field": 1},
				{},
			},
		}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{1001, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{1001, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
		}.Marshal(),
	},
	{
		desc:          "required field in repeated extension message set",
		checkFastInit: true,
		decodeTo: makeMessages(protobuild.Message{
			"multi": []protobuild.Message{
				{"required_field": 1},
				{"required_field": 2},
			},
		}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{1001, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
			}),
			protopack.Tag{1001, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2),
			}),
		}.Marshal(),
	},
	{
		desc: "nil messages",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*test3pb.TestAllTypes)(nil),
			(*testeditionspb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
	},
	{
		desc:    "legacy",
		partial: true,
		decodeTo: makeMessages(protobuild.Message{
			"f1": protobuild.Message{
				"optional_int32":      1,
				"optional_child_enum": "ALPHA",
				"optional_child_message": protobuild.Message{
					"f1": "x",
				},
				"optionalgroup": protobuild.Message{
					"f1": "x",
				},
				"repeated_child_message": []protobuild.Message{
					{"f1": "x"},
				},
				"repeatedgroup": []protobuild.Message{
					{"f1": "x"},
				},
				"map_bool_child_message": map[bool]protobuild.Message{
					true: {"f1": "x"},
				},
				"oneof_child_message": protobuild.Message{
					"f1": "x",
				},
			},
		}, &legacypb.Legacy{}),
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{101, protopack.VarintType}, protopack.Varint(1),
				protopack.Tag{115, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{116, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
				}),
				protopack.Tag{120, protopack.StartGroupType},
				protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
				protopack.Tag{120, protopack.EndGroupType},
				protopack.Tag{516, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
				}),
				protopack.Tag{520, protopack.StartGroupType},
				protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
				protopack.Tag{520, protopack.EndGroupType},
				protopack.Tag{616, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
					protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
						protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
					}),
				}),
				protopack.Tag{716, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.BytesType}, protopack.String("x"),
				}),
			}),
		}.Marshal(),
		validationStatus: impl.ValidationUnknown,
	},
	{
		desc: "first reserved field number",
		decodeTo: makeMessages(protobuild.Message{
			protobuild.Unknown: protopack.Message{
				protopack.Tag{protopack.FirstReservedNumber, protopack.VarintType}, protopack.Varint(1004),
			}.Marshal(),
		}),
		wire: protopack.Message{
			protopack.Tag{protopack.FirstReservedNumber, protopack.VarintType}, protopack.Varint(1004),
		}.Marshal(),
	},
	{
		desc: "last reserved field number",
		decodeTo: makeMessages(protobuild.Message{
			protobuild.Unknown: protopack.Message{
				protopack.Tag{protopack.LastReservedNumber, protopack.VarintType}, protopack.Varint(1005),
			}.Marshal(),
		}),
		wire: protopack.Message{
			protopack.Tag{protopack.LastReservedNumber, protopack.VarintType}, protopack.Varint(1005),
		}.Marshal(),
	},
	{
		desc: "nested unknown extension",
		unmarshalOptions: proto.UnmarshalOptions{
			DiscardUnknown: true,
			Resolver: filterResolver{
				filter: func(name protoreflect.FullName) bool {
					switch name.Name() {
					case "optional_nested_message",
						"optional_int32":
						return true
					}
					return false
				},
				resolver: protoregistry.GlobalTypes,
			},
		},
		decodeTo: makeMessages(protobuild.Message{
			"optional_nested_message": protobuild.Message{
				"corecursive": protobuild.Message{
					"optional_nested_message": protobuild.Message{
						"corecursive": protobuild.Message{
							"optional_int32": 42,
						},
					},
				},
			},
		}, &testpb.TestAllExtensions{}),
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
						protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
							protopack.Tag{1, protopack.VarintType}, protopack.Varint(42),
							protopack.Tag{2, protopack.VarintType}, protopack.Varint(43),
						}),
					}),
				}),
			}),
		}.Marshal(),
	},
}

var testInvalidMessages = []testProto{
	{
		desc: "invalid UTF-8 in optional string field",
		decodeTo: makeMessages(protobuild.Message{
			"optional_string": "abc\xff",
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{14, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in singular string field",
		decodeTo: makeMessages(protobuild.Message{
			"singular_string": "abc\xff",
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{94, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in repeated string field",
		decodeTo: makeMessages(protobuild.Message{
			"repeated_string": []string{"foo", "abc\xff"},
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{44, protopack.BytesType}, protopack.String("foo"),
			protopack.Tag{44, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in nested message",
		decodeTo: makeMessages(protobuild.Message{
			"optional_nested_message": protobuild.Message{
				"corecursive": protobuild.Message{
					"singular_string": "abc\xff",
				},
			},
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{94, protopack.BytesType}, protopack.String("abc\xff"),
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in oneof field",
		decodeTo: makeMessages(protobuild.Message{
			"oneof_string": "abc\xff",
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{protopack.Tag{113, protopack.BytesType}, protopack.String("abc\xff")}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in map key",
		decodeTo: makeMessages(protobuild.Message{
			"map_string_string": map[string]string{"key\xff": "val"},
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{69, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("key\xff"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("val"),
			}),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in map value",
		decodeTo: makeMessages(protobuild.Message{
			"map_string_string": map[string]string{"key": "val\xff"},
		}, &test3pb.TestAllTypes{}),
		wire: protopack.Message{
			protopack.Tag{69, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.BytesType}, protopack.String("key"),
				protopack.Tag{2, protopack.BytesType}, protopack.String("val\xff"),
			}),
		}.Marshal(),
	},
	{
		desc: "invalid field number zero",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testeditionspb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{protopack.MinValidNumber - 1, protopack.VarintType}, protopack.Varint(1001),
		}.Marshal(),
	},
	{
		desc: "invalid field numbers zero and one",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testeditionspb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{protopack.MinValidNumber - 1, protopack.VarintType}, protopack.Varint(1002),
			protopack.Tag{protopack.MinValidNumber, protopack.VarintType}, protopack.Varint(1003),
		}.Marshal(),
	},
	{
		desc: "invalid field numbers max and max+1",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{protopack.MaxValidNumber, protopack.VarintType}, protopack.Varint(1006),
			protopack.Tag{protopack.MaxValidNumber + 1, protopack.VarintType}, protopack.Varint(1007),
		}.Marshal(),
	},
	{
		desc: "invalid field number max+1",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{protopack.MaxValidNumber + 1, protopack.VarintType}, protopack.Varint(1008),
		}.Marshal(),
	},
	{
		desc: "invalid field number wraps int32",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Varint(2234993595104), protopack.Varint(0),
		}.Marshal(),
	},
	{
		desc:     "invalid field number in map",
		decodeTo: []proto.Message{(*testpb.TestAllTypes)(nil)},
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1056),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1156),
				protopack.Tag{protopack.MaxValidNumber + 1, protopack.VarintType}, protopack.Varint(0),
			}),
		}.Marshal(),
	},
	{
		desc: "invalid tag varint",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: []byte{0xff},
	},
	{
		desc: "field number too small",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{0, protopack.VarintType}, protopack.Varint(0),
		}.Marshal(),
	},
	{
		desc: "field number too large",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{protowire.MaxValidNumber + 1, protopack.VarintType}, protopack.Varint(0),
		}.Marshal(),
	},
	{
		desc: "invalid tag varint in message field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Raw{0xff},
			}),
		}.Marshal(),
	},
	{
		desc: "invalid tag varint in repeated message field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{48, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Raw{0xff},
			}),
		}.Marshal(),
	},
	{
		desc: "invalid varint in group field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{16, protopack.StartGroupType},
			protopack.Tag{1000, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Raw{0xff},
			}),
			protopack.Tag{16, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "invalid varint in repeated group field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{46, protopack.StartGroupType},
			protopack.Tag{1001, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Raw{0xff},
			}),
			protopack.Tag{46, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "unterminated repeated group field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{46, protopack.StartGroupType},
		}.Marshal(),
	},
	{
		desc: "invalid tag varint in map item",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{56, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(0),
				protopack.Raw{0xff},
			}),
		}.Marshal(),
	},
	{
		desc: "invalid tag varint in map message value",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{71, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Raw{0xff},
				}),
			}),
		}.Marshal(),
	},
	{
		desc: "invalid packed int32 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{31, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed int64 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{32, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed uint32 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{33, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed uint64 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{34, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed sint32 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{35, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed sint64 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{36, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "invalid packed fixed32 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{37, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed fixed64 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{38, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed sfixed32 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{39, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed sfixed64 field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{40, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed float field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{41, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed double field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{42, protopack.BytesType}, protopack.Bytes{0x00},
		}.Marshal(),
	},
	{
		desc: "invalid packed bool field",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{43, protopack.BytesType}, protopack.Bytes{0xff},
		}.Marshal(),
	},
	{
		desc: "bytes field overruns message",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType}, protopack.LengthPrefix{protopack.Message{
				protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix{protopack.Message{
					protopack.Tag{15, protopack.BytesType}, protopack.Varint(2),
				}},
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(0),
			}},
		}.Marshal(),
	},
	{
		desc: "varint field overruns message",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType},
		}.Marshal(),
	},
	{
		desc: "bytes field lacks size",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{18, protopack.BytesType},
		}.Marshal(),
	},
	{
		desc: "varint overflow",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType},
			protopack.Raw("\xff\xff\xff\xff\xff\xff\xff\xff\xff\x02"),
		}.Marshal(),
	},
	{
		desc: "varint length overrun",
		decodeTo: []proto.Message{
			(*testpb.TestAllTypes)(nil),
			(*testpb.TestAllExtensions)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.VarintType},
			protopack.Raw("\xff\xff\xff\xff\xff\xff\xff\xff\xff"),
		}.Marshal(),
	},
}

type filterResolver struct {
	filter   func(name protoreflect.FullName) bool
	resolver protoregistry.ExtensionTypeResolver
}

func (f filterResolver) FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error) {
	if !f.filter(field) {
		return nil, protoregistry.NotFound
	}
	return f.resolver.FindExtensionByName(field)
}

func (f filterResolver) FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error) {
	xt, err := f.resolver.FindExtensionByNumber(message, field)
	if err != nil {
		return nil, err
	}
	if !f.filter(xt.TypeDescriptor().FullName()) {
		return nil, protoregistry.NotFound
	}
	return xt, nil
}

func roundTripMessage(dst, src proto.Message) error {
	b, err := proto.Marshal(src)
	if err != nil {
		return err
	}
	return proto.Unmarshal(b, dst)
}
