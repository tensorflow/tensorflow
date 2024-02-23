// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protopack

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

var msgDesc = func() protoreflect.MessageDescriptor {
	const s = `
		name:   "test.proto"
		syntax: "proto2"
		message_type: [{
			name: "Message"
			field: [
				{name:"f1"  number:1  label:LABEL_REPEATED type:TYPE_BOOL     options:{packed:true}},
				{name:"f2"  number:2  label:LABEL_REPEATED type:TYPE_INT64    options:{packed:true}},
				{name:"f3"  number:3  label:LABEL_REPEATED type:TYPE_SINT64   options:{packed:true}},
				{name:"f4"  number:4  label:LABEL_REPEATED type:TYPE_UINT64   options:{packed:true}},
				{name:"f5"  number:5  label:LABEL_REPEATED type:TYPE_FIXED32  options:{packed:true}},
				{name:"f6"  number:6  label:LABEL_REPEATED type:TYPE_SFIXED32 options:{packed:true}},
				{name:"f7"  number:7  label:LABEL_REPEATED type:TYPE_FLOAT    options:{packed:true}},
				{name:"f8"  number:8  label:LABEL_REPEATED type:TYPE_FIXED64  options:{packed:true}},
				{name:"f9"  number:9  label:LABEL_REPEATED type:TYPE_SFIXED64 options:{packed:true}},
				{name:"f10" number:10 label:LABEL_REPEATED type:TYPE_DOUBLE   options:{packed:true}},
				{name:"f11" number:11 label:LABEL_OPTIONAL type:TYPE_STRING},
				{name:"f12" number:12 label:LABEL_OPTIONAL type:TYPE_BYTES},
				{name:"f13" number:13 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".Message"},
				{name:"f14" number:14 label:LABEL_OPTIONAL type:TYPE_GROUP   type_name:".Message.F14"}
			]
			nested_type: [{name: "F14"}]
		}]
	`
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(pb, nil)
	if err != nil {
		panic(err)
	}
	return fd.Messages().Get(0)
}()

// dhex decodes a hex-string and returns the bytes and panics if s is invalid.
func dhex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func TestPack(t *testing.T) {
	tests := []struct {
		raw      []byte
		msg      Message
		msgDesc  protoreflect.MessageDescriptor
		inferMsg bool

		wantOutCompact string
		wantOutMulti   string
		wantOutSource  string
	}{{
		raw: dhex("080088808080800002088280808080000a09010002828080808000"),
		msg: Message{
			Tag{1, VarintType}, Bool(false),
			Denormalized{5, Tag{1, VarintType}}, Uvarint(2),
			Tag{1, VarintType}, Denormalized{5, Uvarint(2)},
			Tag{1, BytesType}, LengthPrefix{Bool(true), Bool(false), Uvarint(2), Denormalized{5, Uvarint(2)}},
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{1, protopack.VarintType}, protopack.Bool(false),
	protopack.Denormalized{+5, protopack.Tag{1, protopack.VarintType}}, protopack.Uvarint(2),
	protopack.Tag{1, protopack.VarintType}, protopack.Denormalized{+5, protopack.Uvarint(2)},
	protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix{protopack.Bool(true), protopack.Bool(false), protopack.Uvarint(2), protopack.Denormalized{+5, protopack.Uvarint(2)}},
}`,
	}, {
		raw: dhex("080088808080800002088280808080000a09010002828080808000"),
		msg: Message{
			Tag{1, VarintType}, Uvarint(0),
			Denormalized{5, Tag{1, VarintType}}, Uvarint(2),
			Tag{1, VarintType}, Denormalized{5, Uvarint(2)},
			Tag{1, BytesType}, Bytes(Message{Bool(true), Bool(false), Uvarint(2), Denormalized{5, Uvarint(2)}}.Marshal()),
		},
		inferMsg: true,
	}, {
		raw: dhex("100010828080808000121980808080808080808001ffffffffffffffff7f828080808000"),
		msg: Message{
			Tag{2, VarintType}, Varint(0),
			Tag{2, VarintType}, Denormalized{5, Varint(2)},
			Tag{2, BytesType}, LengthPrefix{Varint(math.MinInt64), Varint(math.MaxInt64), Denormalized{5, Varint(2)}},
		},
		msgDesc:        msgDesc,
		wantOutCompact: `Message{Tag{2, Varint}, Varint(0), Tag{2, Varint}, Denormalized{+5, Varint(2)}, Tag{2, Bytes}, LengthPrefix{Varint(-9223372036854775808), Varint(9223372036854775807), Denormalized{+5, Varint(2)}}}`,
	}, {
		raw: dhex("1801188180808080001a1affffffffffffffffff01feffffffffffffffff01818080808000"),
		msg: Message{
			Tag{3, VarintType}, Svarint(-1),
			Tag{3, VarintType}, Denormalized{5, Svarint(-1)},
			Tag{3, BytesType}, LengthPrefix{Svarint(math.MinInt64), Svarint(math.MaxInt64), Denormalized{5, Svarint(-1)}},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{3, Varint}, Svarint(-1),
	Tag{3, Varint}, Denormalized{+5, Svarint(-1)},
	Tag{3, Bytes}, LengthPrefix{Svarint(-9223372036854775808), Svarint(9223372036854775807), Denormalized{+5, Svarint(-1)}},
}`,
	}, {
		raw: dhex("200120818080808000221100ffffffffffffffffff01818080808000"),
		msg: Message{
			Tag{4, VarintType}, Uvarint(+1),
			Tag{4, VarintType}, Denormalized{5, Uvarint(+1)},
			Tag{4, BytesType}, LengthPrefix{Uvarint(0), Uvarint(math.MaxUint64), Denormalized{5, Uvarint(+1)}},
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{4, protopack.VarintType}, protopack.Uvarint(1),
	protopack.Tag{4, protopack.VarintType}, protopack.Denormalized{+5, protopack.Uvarint(1)},
	protopack.Tag{4, protopack.BytesType}, protopack.LengthPrefix{protopack.Uvarint(0), protopack.Uvarint(18446744073709551615), protopack.Denormalized{+5, protopack.Uvarint(1)}},
}`,
	}, {
		raw: dhex("2d010000002a0800000000ffffffff"),
		msg: Message{
			Tag{5, Fixed32Type}, Uint32(+1),
			Tag{5, BytesType}, LengthPrefix{Uint32(0), Uint32(math.MaxUint32)},
		},
		msgDesc:        msgDesc,
		wantOutCompact: `Message{Tag{5, Fixed32}, Uint32(1), Tag{5, Bytes}, LengthPrefix{Uint32(0), Uint32(4294967295)}}`,
	}, {
		raw: dhex("35ffffffff320800000080ffffff7f"),
		msg: Message{
			Tag{6, Fixed32Type}, Int32(-1),
			Tag{6, BytesType}, LengthPrefix{Int32(math.MinInt32), Int32(math.MaxInt32)},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{6, Fixed32}, Int32(-1),
	Tag{6, Bytes}, LengthPrefix{Int32(-2147483648), Int32(2147483647)},
}`,
	}, {
		raw: dhex("3ddb0f49403a1001000000ffff7f7f0000807f000080ff"),
		msg: Message{
			Tag{7, Fixed32Type}, Float32(math.Pi),
			Tag{7, BytesType}, LengthPrefix{Float32(math.SmallestNonzeroFloat32), Float32(math.MaxFloat32), Float32(math.Inf(+1)), Float32(math.Inf(-1))},
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{7, protopack.Fixed32Type}, protopack.Float32(3.1415927),
	protopack.Tag{7, protopack.BytesType}, protopack.LengthPrefix{protopack.Float32(1e-45), protopack.Float32(3.4028235e+38), protopack.Float32(math.Inf(+1)), protopack.Float32(math.Inf(-1))},
}`,
	}, {
		raw: dhex("41010000000000000042100000000000000000ffffffffffffffff"),
		msg: Message{
			Tag{8, Fixed64Type}, Uint64(+1),
			Tag{8, BytesType}, LengthPrefix{Uint64(0), Uint64(math.MaxUint64)},
		},
		msgDesc:        msgDesc,
		wantOutCompact: `Message{Tag{8, Fixed64}, Uint64(1), Tag{8, Bytes}, LengthPrefix{Uint64(0), Uint64(18446744073709551615)}}`,
	}, {
		raw: dhex("49ffffffffffffffff4a100000000000000080ffffffffffffff7f"),
		msg: Message{
			Tag{9, Fixed64Type}, Int64(-1),
			Tag{9, BytesType}, LengthPrefix{Int64(math.MinInt64), Int64(math.MaxInt64)},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{9, Fixed64}, Int64(-1),
	Tag{9, Bytes}, LengthPrefix{Int64(-9223372036854775808), Int64(9223372036854775807)},
}`,
	}, {
		raw: dhex("51182d4454fb21094052200100000000000000ffffffffffffef7f000000000000f07f000000000000f0ff"),
		msg: Message{
			Tag{10, Fixed64Type}, Float64(math.Pi),
			Tag{10, BytesType}, LengthPrefix{Float64(math.SmallestNonzeroFloat64), Float64(math.MaxFloat64), Float64(math.Inf(+1)), Float64(math.Inf(-1))},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{10, Fixed64}, Float64(3.141592653589793),
	Tag{10, Bytes}, LengthPrefix{Float64(5e-324), Float64(1.7976931348623157e+308), Float64(+Inf), Float64(-Inf)},
}`,
	}, {
		raw: dhex("5a06737472696e675a868080808000737472696e67"),
		msg: Message{
			Tag{11, BytesType}, String("string"),
			Tag{11, BytesType}, Denormalized{+5, String("string")},
		},
		msgDesc:        msgDesc,
		wantOutCompact: `Message{Tag{11, Bytes}, String("string"), Tag{11, Bytes}, Denormalized{+5, String("string")}}`,
	}, {
		raw: dhex("62056279746573628580808080006279746573"),
		msg: Message{
			Tag{12, BytesType}, Bytes("bytes"),
			Tag{12, BytesType}, Denormalized{+5, Bytes("bytes")},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{12, Bytes}, Bytes("bytes"),
	Tag{12, Bytes}, Denormalized{+5, Bytes("bytes")},
}`,
	}, {
		raw: dhex("6a28a006ffffffffffffffffff01a506ffffffffa106ffffffffffffffffa206056279746573a306a406"),
		msg: Message{
			Tag{13, BytesType}, LengthPrefix(Message{
				Tag{100, VarintType}, Uvarint(math.MaxUint64),
				Tag{100, Fixed32Type}, Uint32(math.MaxUint32),
				Tag{100, Fixed64Type}, Uint64(math.MaxUint64),
				Tag{100, BytesType}, Bytes("bytes"),
				Tag{100, StartGroupType}, Tag{100, EndGroupType},
			}),
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{13, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
		protopack.Tag{100, protopack.VarintType}, protopack.Uvarint(18446744073709551615),
		protopack.Tag{100, protopack.Fixed32Type}, protopack.Uint32(4294967295),
		protopack.Tag{100, protopack.Fixed64Type}, protopack.Uint64(18446744073709551615),
		protopack.Tag{100, protopack.BytesType}, protopack.Bytes("bytes"),
		protopack.Tag{100, protopack.StartGroupType},
		protopack.Tag{100, protopack.EndGroupType},
	}),
}`,
	}, {
		raw: dhex("6a28a006ffffffffffffffffff01a506ffffffffa106ffffffffffffffffa206056279746573a306a406"),
		msg: Message{
			Tag{13, BytesType}, LengthPrefix(Message{
				Tag{100, VarintType}, Uvarint(math.MaxUint64),
				Tag{100, Fixed32Type}, Uint32(math.MaxUint32),
				Tag{100, Fixed64Type}, Uint64(math.MaxUint64),
				Tag{100, BytesType}, Bytes("bytes"),
				Tag{100, StartGroupType}, Tag{100, EndGroupType},
			}),
		},
		inferMsg: true,
	}, {
		raw: dhex("6a28a006ffffffffffffffffff01a506ffffffffa106ffffffffffffffffa206056279746573a306ac06"),
		msg: Message{
			Tag{13, BytesType}, Bytes(Message{
				Tag{100, VarintType}, Uvarint(math.MaxUint64),
				Tag{100, Fixed32Type}, Uint32(math.MaxUint32),
				Tag{100, Fixed64Type}, Uint64(math.MaxUint64),
				Tag{100, BytesType}, Bytes("bytes"),
				Tag{100, StartGroupType}, Tag{101, EndGroupType},
			}.Marshal()),
		},
		inferMsg: true,
	}, {
		raw: dhex("6aa88080808000a006ffffffffffffffffff01a506ffffffffa106ffffffffffffffffa206056279746573a306a406"),
		msg: Message{
			Tag{13, BytesType}, Denormalized{5, LengthPrefix(Message{
				Tag{100, VarintType}, Uvarint(math.MaxUint64),
				Tag{100, Fixed32Type}, Uint32(math.MaxUint32),
				Tag{100, Fixed64Type}, Uint64(math.MaxUint64),
				Tag{100, BytesType}, Bytes("bytes"),
				Tag{100, StartGroupType}, Tag{100, EndGroupType},
			})},
		},
		msgDesc:        msgDesc,
		wantOutCompact: `Message{Tag{13, Bytes}, Denormalized{+5, LengthPrefix(Message{Tag{100, Varint}, Uvarint(18446744073709551615), Tag{100, Fixed32}, Uint32(4294967295), Tag{100, Fixed64}, Uint64(18446744073709551615), Tag{100, Bytes}, Bytes("bytes"), Tag{100, StartGroup}, Tag{100, EndGroup}})}}`,
	}, {
		raw: dhex("73a006ffffffffffffffffff01a506ffffffffa106ffffffffffffffffa206056279746573a306a40674"),
		msg: Message{
			Tag{14, StartGroupType}, Message{
				Tag{100, VarintType}, Uvarint(math.MaxUint64),
				Tag{100, Fixed32Type}, Uint32(math.MaxUint32),
				Tag{100, Fixed64Type}, Uint64(math.MaxUint64),
				Tag{100, BytesType}, Bytes("bytes"),
				Tag{100, StartGroupType}, Tag{100, EndGroupType},
			},
			Tag{14, EndGroupType},
		},
		msgDesc: msgDesc,
		wantOutMulti: `Message{
	Tag{14, StartGroup},
	Message{
		Tag{100, Varint}, Uvarint(18446744073709551615),
		Tag{100, Fixed32}, Uint32(4294967295),
		Tag{100, Fixed64}, Uint64(18446744073709551615),
		Tag{100, Bytes}, Bytes("bytes"),
		Tag{100, StartGroup},
		Tag{100, EndGroup},
	},
	Tag{14, EndGroup},
}`,
	}, {
		raw: dhex("d0faa972cd02a5f09051c2d8aa0d6a26a89c311eddef024b423c0f6f47b64227a1600db56e3f73d4113096c9a88e2b99f2d847516853d76a1a6e9811c85a2ab3"),
		msg: Message{
			Tag{29970346, VarintType}, Uvarint(333),
			Tag{21268228, Fixed32Type}, Uint32(229300418),
			Tag{13, BytesType}, LengthPrefix(Message{
				Tag{100805, VarintType}, Uvarint(30),
				Tag{5883, Fixed32Type}, Uint32(255607371),
				Tag{13, Type(7)},
				Raw("G\xb6B'\xa1`\r\xb5n?s\xd4\x110\x96ɨ\x8e+\x99\xf2\xd8GQhS"),
			}),
			Tag{1706, Type(7)},
			Raw("\x1an\x98\x11\xc8Z*\xb3"),
		},
		msgDesc: msgDesc,
	}, {
		raw: dhex("3d08d0e57f"),
		msg: Message{
			Tag{7, Fixed32Type}, Float32(math.Float32frombits(
				// TODO: Remove workaround for compiler bug (see https://golang.org/issues/27193).
				func() uint32 { return 0x7fe5d008 }(),
			)),
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{7, protopack.Fixed32Type}, protopack.Float32(math.Float32frombits(0x7fe5d008)),
}`,
	}, {
		raw: dhex("51a8d65110771bf97f"),
		msg: Message{
			Tag{10, Fixed64Type}, Float64(math.Float64frombits(0x7ff91b771051d6a8)),
		},
		msgDesc: msgDesc,
		wantOutSource: `protopack.Message{
	protopack.Tag{10, protopack.Fixed64Type}, protopack.Float64(math.Float64frombits(0x7ff91b771051d6a8)),
}`,
	}, {
		raw: dhex("ab2c14481ab3e9a76d937fb4dd5e6c616ef311f62b7fe888785fca5609ffe81c1064e50dd7a9edb408d317e2891c0d54c719446938d41ab0ccf8e61dc28b0ebb"),
		msg: Message{
			Tag{709, StartGroupType},
			Tag{2, EndGroupType},
			Tag{9, VarintType}, Uvarint(26),
			Tag{28655254, StartGroupType},
			Message{
				Tag{2034, StartGroupType},
				Tag{194006, EndGroupType},
			},
			Tag{13, EndGroupType},
			Tag{12, Fixed64Type}, Uint64(9865274810543764334),
			Tag{15, VarintType}, Uvarint(95),
			Tag{1385, BytesType}, Bytes("\xff\xe8\x1c\x10d\xe5\rש"),
			Tag{17229, Fixed32Type}, Uint32(2313295827),
			Tag{3, EndGroupType},
			Tag{1, Fixed32Type}, Uint32(1142540116),
			Tag{13, Fixed64Type}, Uint64(2154683029754926136),
			Tag{28856, BytesType},
			Raw("\xbb"),
		},
		msgDesc: msgDesc,
	}, {
		raw: dhex("29baa4ac1c1e0a20183393bac434b8d3559337ec940050038770eaa9937f98e4"),
		msg: Message{
			Tag{5, Fixed64Type}, Uint64(1738400580611384506),
			Tag{6, StartGroupType},
			Message{
				Tag{13771682, StartGroupType},
				Message{
					Tag{175415, VarintType}, Uvarint(7059),
				},
				Denormalized{+1, Tag{333, EndGroupType}},
				Tag{10, VarintType}, Uvarint(3),
				Tag{1792, Type(7)},
				Raw("꩓\u007f\x98\xe4"),
			},
		},
		msgDesc: msgDesc,
	}}

	equateFloatBits := cmp.Options{
		cmp.Comparer(func(x, y Float32) bool {
			return math.Float32bits(float32(x)) == math.Float32bits(float32(y))
		}),
		cmp.Comparer(func(x, y Float64) bool {
			return math.Float64bits(float64(x)) == math.Float64bits(float64(y))
		}),
	}
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var msg Message
			raw := tt.msg.Marshal()
			msg.unmarshal(tt.raw, tt.msgDesc, tt.inferMsg)

			if !bytes.Equal(raw, tt.raw) {
				t.Errorf("Marshal() mismatch:\ngot  %x\nwant %x", raw, tt.raw)
			}
			if diff := cmp.Diff(tt.msg, msg, equateFloatBits); diff != "" {
				t.Errorf("Unmarshal() mismatch (-want +got):\n%s", diff)
			}
			if got, want := tt.msg.Size(), len(tt.raw); got != want {
				t.Errorf("Size() = %v, want %v", got, want)
			}
			if tt.wantOutCompact != "" {
				gotOut := fmt.Sprintf("%v", tt.msg)
				if string(gotOut) != tt.wantOutCompact {
					t.Errorf("fmt.Sprintf(%q, msg):\ngot:  %s\nwant: %s", "%v", gotOut, tt.wantOutCompact)
				}
			}
			if tt.wantOutMulti != "" {
				gotOut := fmt.Sprintf("%+v", tt.msg)
				if string(gotOut) != tt.wantOutMulti {
					t.Errorf("fmt.Sprintf(%q, msg):\ngot:  %s\nwant: %s", "%+v", gotOut, tt.wantOutMulti)
				}
			}
			if tt.wantOutSource != "" {
				gotOut := fmt.Sprintf("%#v", tt.msg)
				if string(gotOut) != tt.wantOutSource {
					t.Errorf("fmt.Sprintf(%q, msg):\ngot:  %s\nwant: %s", "%#v", gotOut, tt.wantOutSource)
				}
			}
		})
	}
}
