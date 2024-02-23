// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"reflect"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoimpl"
	"google.golang.org/protobuf/testing/protopack"

	"google.golang.org/protobuf/types/descriptorpb"
)

func init() {
	if flags.ProtoLegacy {
		testValidMessages = append(testValidMessages, noEnforceUTF8TestProtos...)
	} else {
		testInvalidMessages = append(testInvalidMessages, noEnforceUTF8TestProtos...)
	}
}

var noEnforceUTF8TestProtos = []testProto{
	{
		desc: "invalid UTF-8 in optional string field",
		decodeTo: []proto.Message{&TestNoEnforceUTF8{
			OptionalString: string("abc\xff"),
		}},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in optional string field of Go bytes",
		decodeTo: []proto.Message{&TestNoEnforceUTF8{
			OptionalBytes: []byte("abc\xff"),
		}},
		wire: protopack.Message{
			protopack.Tag{2, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in repeated string field",
		decodeTo: []proto.Message{&TestNoEnforceUTF8{
			RepeatedString: []string{string("foo"), string("abc\xff")},
		}},
		wire: protopack.Message{
			protopack.Tag{3, protopack.BytesType}, protopack.String("foo"),
			protopack.Tag{3, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in repeated string field of Go bytes",
		decodeTo: []proto.Message{&TestNoEnforceUTF8{
			RepeatedBytes: [][]byte{[]byte("foo"), []byte("abc\xff")},
		}},
		wire: protopack.Message{
			protopack.Tag{4, protopack.BytesType}, protopack.String("foo"),
			protopack.Tag{4, protopack.BytesType}, protopack.String("abc\xff"),
		}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in oneof string field",
		decodeTo: []proto.Message{
			&TestNoEnforceUTF8{OneofField: &TestNoEnforceUTF8_OneofString{string("abc\xff")}},
		},
		wire: protopack.Message{protopack.Tag{5, protopack.BytesType}, protopack.String("abc\xff")}.Marshal(),
	},
	{
		desc: "invalid UTF-8 in oneof string field of Go bytes",
		decodeTo: []proto.Message{
			&TestNoEnforceUTF8{OneofField: &TestNoEnforceUTF8_OneofBytes{[]byte("abc\xff")}},
		},
		wire: protopack.Message{protopack.Tag{6, protopack.BytesType}, protopack.String("abc\xff")}.Marshal(),
	},
}

type TestNoEnforceUTF8 struct {
	OptionalString string       `protobuf:"bytes,1,opt,name=optional_string"`
	OptionalBytes  []byte       `protobuf:"bytes,2,opt,name=optional_bytes"`
	RepeatedString []string     `protobuf:"bytes,3,rep,name=repeated_string"`
	RepeatedBytes  [][]byte     `protobuf:"bytes,4,rep,name=repeated_bytes"`
	OneofField     isOneofField `protobuf_oneof:"oneof_field"`
}

type isOneofField interface{ isOneofField() }

type TestNoEnforceUTF8_OneofString struct {
	OneofString string `protobuf:"bytes,5,opt,name=oneof_string,oneof"`
}
type TestNoEnforceUTF8_OneofBytes struct {
	OneofBytes []byte `protobuf:"bytes,6,opt,name=oneof_bytes,oneof"`
}

func (*TestNoEnforceUTF8_OneofString) isOneofField() {}
func (*TestNoEnforceUTF8_OneofBytes) isOneofField()  {}

func (m *TestNoEnforceUTF8) ProtoReflect() protoreflect.Message {
	return messageInfo_TestNoEnforceUTF8.MessageOf(m)
}

var messageInfo_TestNoEnforceUTF8 = protoimpl.MessageInfo{
	GoReflectType: reflect.TypeOf((*TestNoEnforceUTF8)(nil)),
	Desc: func() protoreflect.MessageDescriptor {
		pb := new(descriptorpb.FileDescriptorProto)
		if err := prototext.Unmarshal([]byte(`
				syntax:  "proto3"
				name:    "test.proto"
				message_type: [{
					name: "TestNoEnforceUTF8"
					field: [
						{name:"optional_string" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
						{name:"optional_bytes"  number:2 label:LABEL_OPTIONAL type:TYPE_STRING},
						{name:"repeated_string" number:3 label:LABEL_REPEATED type:TYPE_STRING},
						{name:"repeated_bytes"  number:4 label:LABEL_REPEATED type:TYPE_STRING},
						{name:"oneof_string"    number:5 label:LABEL_OPTIONAL type:TYPE_STRING, oneof_index:0},
						{name:"oneof_bytes"     number:6 label:LABEL_OPTIONAL type:TYPE_STRING, oneof_index:0}
					]
					oneof_decl: [{name:"oneof_field"}]
				}]
			`), pb); err != nil {
			panic(err)
		}
		fd, err := protodesc.NewFile(pb, nil)
		if err != nil {
			panic(err)
		}
		md := fd.Messages().Get(0)
		for i := 0; i < md.Fields().Len(); i++ {
			md.Fields().Get(i).(*filedesc.Field).L1.HasEnforceUTF8 = true
			md.Fields().Get(i).(*filedesc.Field).L1.EnforceUTF8 = false
		}
		return md
	}(),
	OneofWrappers: []interface{}{
		(*TestNoEnforceUTF8_OneofString)(nil),
		(*TestNoEnforceUTF8_OneofBytes)(nil),
	},
}
