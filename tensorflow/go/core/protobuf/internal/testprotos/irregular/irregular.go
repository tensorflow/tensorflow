// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package irregular

import (
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"

	"google.golang.org/protobuf/types/descriptorpb"
)

type IrregularMessage struct {
	set   bool
	value string
}

func (m *IrregularMessage) ProtoReflect() protoreflect.Message { return (*message)(m) }

type message IrregularMessage

type messageType struct{}

func (messageType) New() protoreflect.Message                  { return &message{} }
func (messageType) Zero() protoreflect.Message                 { return (*message)(nil) }
func (messageType) Descriptor() protoreflect.MessageDescriptor { return fileDesc.Messages().Get(0) }

func (m *message) New() protoreflect.Message                  { return &message{} }
func (m *message) Descriptor() protoreflect.MessageDescriptor { return fileDesc.Messages().Get(0) }
func (m *message) Type() protoreflect.MessageType             { return messageType{} }
func (m *message) Interface() protoreflect.ProtoMessage       { return (*IrregularMessage)(m) }
func (m *message) ProtoMethods() *protoiface.Methods          { return nil }

var fieldDescS = fileDesc.Messages().Get(0).Fields().Get(0)

func (m *message) Range(f func(protoreflect.FieldDescriptor, protoreflect.Value) bool) {
	if m.set {
		f(fieldDescS, protoreflect.ValueOf(m.value))
	}
}

func (m *message) Has(fd protoreflect.FieldDescriptor) bool {
	if fd == fieldDescS {
		return m.set
	}
	panic("invalid field descriptor")
}

func (m *message) Clear(fd protoreflect.FieldDescriptor) {
	if fd == fieldDescS {
		m.value = ""
		m.set = false
		return
	}
	panic("invalid field descriptor")
}

func (m *message) Get(fd protoreflect.FieldDescriptor) protoreflect.Value {
	if fd == fieldDescS {
		return protoreflect.ValueOf(m.value)
	}
	panic("invalid field descriptor")
}

func (m *message) Set(fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	if fd == fieldDescS {
		m.value = v.String()
		m.set = true
		return
	}
	panic("invalid field descriptor")
}

func (m *message) Mutable(protoreflect.FieldDescriptor) protoreflect.Value {
	panic("invalid field descriptor")
}

func (m *message) NewField(protoreflect.FieldDescriptor) protoreflect.Value {
	panic("invalid field descriptor")
}

func (m *message) WhichOneof(protoreflect.OneofDescriptor) protoreflect.FieldDescriptor {
	panic("invalid oneof descriptor")
}

func (m *message) GetUnknown() protoreflect.RawFields { return nil }
func (m *message) SetUnknown(protoreflect.RawFields)  { return }

func (m *message) IsValid() bool {
	return m != nil
}

var fileDesc = func() protoreflect.FileDescriptor {
	p := &descriptorpb.FileDescriptorProto{}
	if err := prototext.Unmarshal([]byte(descriptorText), p); err != nil {
		panic(err)
	}
	file, err := protodesc.NewFile(p, nil)
	if err != nil {
		panic(err)
	}
	return file
}()

func file_internal_testprotos_irregular_irregular_proto_init() { _ = fileDesc }

const descriptorText = `
  name: "internal/testprotos/irregular/irregular.proto"
  package: "goproto.proto.thirdparty"
  message_type {
    name: "IrregularMessage"
    field {
      name: "s"
      number: 1
      label: LABEL_OPTIONAL
      type: TYPE_STRING
      json_name: "s"
    }
  }
  options {
    go_package: "google.golang.org/protobuf/internal/testprotos/irregular"
  }
`

type AberrantMessage int

func (m AberrantMessage) ProtoMessage()            {}
func (m AberrantMessage) Reset()                   {}
func (m AberrantMessage) String() string           { return "" }
func (m AberrantMessage) Marshal() ([]byte, error) { return nil, nil }
func (m AberrantMessage) Unmarshal([]byte) error   { return nil }
