// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocmp

import (
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/testing/protopack"
	"google.golang.org/protobuf/types/known/anypb"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

func init() {
	detrand.Disable()
}

func TestTransform(t *testing.T) {
	tests := []struct {
		in   proto.Message
		want Message
	}{{
		in: &testpb.TestAllTypes{
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
		},
		want: Message{
			messageTypeKey:            messageMetaOf(&testpb.TestAllTypes{}),
			"optional_bool":           bool(false),
			"optional_int32":          int32(-32),
			"optional_int64":          int64(-64),
			"optional_uint32":         uint32(32),
			"optional_uint64":         uint64(64),
			"optional_float":          float32(32.32),
			"optional_double":         float64(64.64),
			"optional_string":         string("string"),
			"optional_bytes":          []byte("bytes"),
			"optional_nested_enum":    enumOf(testpb.TestAllTypes_NEG),
			"optional_nested_message": Message{messageTypeKey: messageMetaOf(&testpb.TestAllTypes_NestedMessage{}), "a": int32(5)},
		},
	}, {
		in: &testpb.TestAllTypes{
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
		},
		want: Message{
			messageTypeKey:    messageMetaOf(&testpb.TestAllTypes{}),
			"repeated_bool":   []bool{false, true},
			"repeated_int32":  []int32{32, -32},
			"repeated_int64":  []int64{64, -64},
			"repeated_uint32": []uint32{0, 32},
			"repeated_uint64": []uint64{0, 64},
			"repeated_float":  []float32{0, 32.32},
			"repeated_double": []float64{0, 64.64},
			"repeated_string": []string{"s1", "s2"},
			"repeated_bytes":  [][]byte{{1}, {2}},
			"repeated_nested_enum": []Enum{
				enumOf(testpb.TestAllTypes_FOO),
				enumOf(testpb.TestAllTypes_BAR),
			},
			"repeated_nested_message": []Message{
				{messageTypeKey: messageMetaOf(&testpb.TestAllTypes_NestedMessage{}), "a": int32(5)},
				{messageTypeKey: messageMetaOf(&testpb.TestAllTypes_NestedMessage{}), "a": int32(-5)},
			},
		},
	}, {
		in: &testpb.TestAllTypes{
			MapBoolBool:     map[bool]bool{true: false},
			MapInt32Int32:   map[int32]int32{-32: 32},
			MapInt64Int64:   map[int64]int64{-64: 64},
			MapUint32Uint32: map[uint32]uint32{0: 32},
			MapUint64Uint64: map[uint64]uint64{0: 64},
			MapInt32Float:   map[int32]float32{32: 32.32},
			MapInt32Double:  map[int32]float64{64: 64.64},
			MapStringString: map[string]string{"k": "v", "empty": ""},
			MapStringBytes:  map[string][]byte{"k": []byte("v"), "empty": nil},
			MapStringNestedEnum: map[string]testpb.TestAllTypes_NestedEnum{
				"k": testpb.TestAllTypes_FOO,
			},
			MapStringNestedMessage: map[string]*testpb.TestAllTypes_NestedMessage{
				"k": {A: proto.Int32(5)},
			},
		},
		want: Message{
			messageTypeKey:      messageMetaOf(&testpb.TestAllTypes{}),
			"map_bool_bool":     map[bool]bool{true: false},
			"map_int32_int32":   map[int32]int32{-32: 32},
			"map_int64_int64":   map[int64]int64{-64: 64},
			"map_uint32_uint32": map[uint32]uint32{0: 32},
			"map_uint64_uint64": map[uint64]uint64{0: 64},
			"map_int32_float":   map[int32]float32{32: 32.32},
			"map_int32_double":  map[int32]float64{64: 64.64},
			"map_string_string": map[string]string{"k": "v", "empty": ""},
			"map_string_bytes":  map[string][]byte{"k": []byte("v"), "empty": []byte{}},
			"map_string_nested_enum": map[string]Enum{
				"k": enumOf(testpb.TestAllTypes_FOO),
			},
			"map_string_nested_message": map[string]Message{
				"k": {messageTypeKey: messageMetaOf(&testpb.TestAllTypes_NestedMessage{}), "a": int32(5)},
			},
		},
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
		want: Message{
			messageTypeKey:                                 messageMetaOf(&testpb.TestAllExtensions{}),
			"[goproto.proto.test.optional_bool]":           bool(false),
			"[goproto.proto.test.optional_int32]":          int32(-32),
			"[goproto.proto.test.optional_int64]":          int64(-64),
			"[goproto.proto.test.optional_uint32]":         uint32(32),
			"[goproto.proto.test.optional_uint64]":         uint64(64),
			"[goproto.proto.test.optional_float]":          float32(32.32),
			"[goproto.proto.test.optional_double]":         float64(64.64),
			"[goproto.proto.test.optional_string]":         string("string"),
			"[goproto.proto.test.optional_bytes]":          []byte("bytes"),
			"[goproto.proto.test.optional_nested_enum]":    enumOf(testpb.TestAllTypes_NEG),
			"[goproto.proto.test.optional_nested_message]": Message{messageTypeKey: messageMetaOf(&testpb.TestAllExtensions_NestedMessage{}), "a": int32(5)},
		},
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
		want: Message{
			messageTypeKey:                         messageMetaOf(&testpb.TestAllExtensions{}),
			"[goproto.proto.test.repeated_bool]":   []bool{false, true},
			"[goproto.proto.test.repeated_int32]":  []int32{32, -32},
			"[goproto.proto.test.repeated_int64]":  []int64{64, -64},
			"[goproto.proto.test.repeated_uint32]": []uint32{0, 32},
			"[goproto.proto.test.repeated_uint64]": []uint64{0, 64},
			"[goproto.proto.test.repeated_float]":  []float32{0, 32.32},
			"[goproto.proto.test.repeated_double]": []float64{0, 64.64},
			"[goproto.proto.test.repeated_string]": []string{"s1", "s2"},
			"[goproto.proto.test.repeated_bytes]":  [][]byte{{1}, {2}},
			"[goproto.proto.test.repeated_nested_enum]": []Enum{
				enumOf(testpb.TestAllTypes_FOO),
				enumOf(testpb.TestAllTypes_BAR),
			},
			"[goproto.proto.test.repeated_nested_message]": []Message{
				{messageTypeKey: messageMetaOf(&testpb.TestAllExtensions_NestedMessage{}), "a": int32(5)},
				{messageTypeKey: messageMetaOf(&testpb.TestAllExtensions_NestedMessage{}), "a": int32(-5)},
			},
		},
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
		want: Message{
			messageTypeKey: messageMetaOf(&testpb.TestAllTypes{}),
			"50000":        protoreflect.RawFields(protopack.Message{protopack.Tag{Number: 50000, Type: protopack.VarintType}, protopack.Uvarint(100)}.Marshal()),
			"50001":        protoreflect.RawFields(protopack.Message{protopack.Tag{Number: 50001, Type: protopack.Fixed32Type}, protopack.Uint32(200)}.Marshal()),
			"50002":        protoreflect.RawFields(protopack.Message{protopack.Tag{Number: 50002, Type: protopack.Fixed64Type}, protopack.Uint64(300)}.Marshal()),
			"50003":        protoreflect.RawFields(protopack.Message{protopack.Tag{Number: 50003, Type: protopack.BytesType}, protopack.String("hello")}.Marshal()),
			"50004": protoreflect.RawFields(protopack.Message{
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
			}.Marshal()),
		},
	}}
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := newTransformer().transformMessage(tt.in.ProtoReflect())
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Transform() mismatch (-want +got):\n%v", diff)
			}
			if got.Unwrap() != tt.in {
				t.Errorf("got.Unwrap() = %p, want %p", got.Unwrap(), tt.in)
			}
		})
	}

	t.Run("messageTypeResolver", func(t *testing.T) {
		r := unaryMessageTypeResolver{
			Type: (&testpb.TestAllTypes{}).ProtoReflect().Type(),
		}
		m := &testpb.TestAllTypes{OptionalBool: proto.Bool(true)}
		in, err := anypb.New(m)
		if err != nil {
			t.Fatalf("anypb.New() failed: %v", err)
		}
		in.TypeUrl = "type.googleapis.com/MagicTestMessage"

		got := newTransformer(MessageTypeResolver(r)).transformMessage(in.ProtoReflect())
		want := Message{
			messageTypeKey: messageMetaOf(&anypb.Any{}),
			"type_url":     "type.googleapis.com/MagicTestMessage",
			"value": Message{
				messageTypeKey:  messageMetaOf(m),
				"optional_bool": true,
			},
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("Transform() mismatch (-want +got):\n%v", diff)
		}
		if got.Unwrap() != in {
			t.Errorf("got.Unwrap() = %p, want %p", got.Unwrap(), in)
		}
	})
}

func enumOf(e protoreflect.Enum) Enum {
	return Enum{e.Number(), e.Descriptor()}
}

func messageMetaOf(m protoreflect.ProtoMessage) messageMeta {
	return messageMeta{m: m, md: m.ProtoReflect().Descriptor()}
}

// A unaryMessageTypeResolver can only resolve one type, and it's
// called "MagicTestMessage".
type unaryMessageTypeResolver struct {
	Type protoreflect.MessageType
}

func (r unaryMessageTypeResolver) FindMessageByName(message protoreflect.FullName) (protoreflect.MessageType, error) {
	if message != "MagicTestMessage" {
		return nil, protoregistry.NotFound
	}
	return r.Type, nil
}

func (r unaryMessageTypeResolver) FindMessageByURL(url string) (protoreflect.MessageType, error) {
	const prefix = "type.googleapis.com/"

	if !strings.HasPrefix(url, prefix) {
		return nil, protoregistry.NotFound
	}
	return r.FindMessageByName(protoreflect.FullName(strings.TrimPrefix(url, prefix)))
}
