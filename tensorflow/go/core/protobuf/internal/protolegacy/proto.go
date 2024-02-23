// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protolegacy is a stub version of the v1 proto package
// to satisfy internal/testprotos/legacy dependencies.
package protolegacy

import (
	"bytes"
	"compress/gzip"
	"errors"
	"fmt"
	"io/ioutil"

	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
	"google.golang.org/protobuf/runtime/protoimpl"
)

const (
	ProtoPackageIsVersion1 = true
	ProtoPackageIsVersion2 = true
	ProtoPackageIsVersion3 = true
)

const (
	WireVarint     = 0
	WireFixed32    = 5
	WireFixed64    = 1
	WireBytes      = 2
	WireStartGroup = 3
	WireEndGroup   = 4
)

type (
	Message                = protoiface.MessageV1
	ExtensionRange         = protoiface.ExtensionRangeV1
	ExtensionDesc          = protoimpl.ExtensionInfo
	Extension              = protoimpl.ExtensionFieldV1
	XXX_InternalExtensions = protoimpl.ExtensionFields
)

func RegisterFile(s string, d []byte) {
	// Decompress the descriptor.
	zr, err := gzip.NewReader(bytes.NewReader(d))
	if err != nil {
		panic(fmt.Sprintf("proto: invalid compressed file descriptor: %v", err))
	}
	b, err := ioutil.ReadAll(zr)
	if err != nil {
		panic(fmt.Sprintf("proto: invalid compressed file descriptor: %v", err))
	}

	// Construct a protoreflect.FileDescriptor from the raw descriptor.
	// Note that DescBuilder.Build automatically registers the constructed
	// file descriptor with the v2 registry.
	protoimpl.DescBuilder{RawDescriptor: b}.Build()
}

func RegisterType(m Message, s string) {
	mt := protoimpl.X.LegacyMessageTypeOf(m, protoreflect.FullName(s))
	if err := protoregistry.GlobalTypes.RegisterMessage(mt); err != nil {
		panic(err)
	}
}

func RegisterMapType(interface{}, string) {
	// Do nothing.
}

func RegisterEnum(string, map[int32]string, map[string]int32) {
	// Do nothing.
}

func RegisterExtension(d *ExtensionDesc) {
	if err := protoregistry.GlobalTypes.RegisterExtension(d); err != nil {
		panic(err)
	}
}

var ErrInternalBadWireType = errors.New("not implemented")

func Size(Message) int                { panic("not implemented") }
func Marshal(Message) ([]byte, error) { panic("not implemented") }
func Unmarshal([]byte, Message) error { panic("not implemented") }

func SizeVarint(uint64) int             { panic("not implemented") }
func EncodeVarint(uint64) []byte        { panic("not implemented") }
func DecodeVarint([]byte) (uint64, int) { panic("not implemented") }

func CompactTextString(Message) string                                  { panic("not implemented") }
func EnumName(map[int32]string, int32) string                           { panic("not implemented") }
func UnmarshalJSONEnum(map[string]int32, []byte, string) (int32, error) { panic("not implemented") }

type Buffer struct{}

func (*Buffer) DecodeFixed32() (uint64, error)      { panic("not implemented") }
func (*Buffer) DecodeFixed64() (uint64, error)      { panic("not implemented") }
func (*Buffer) DecodeGroup(Message) error           { panic("not implemented") }
func (*Buffer) DecodeMessage(Message) error         { panic("not implemented") }
func (*Buffer) DecodeRawBytes(bool) ([]byte, error) { panic("not implemented") }
func (*Buffer) DecodeStringBytes() (string, error)  { panic("not implemented") }
func (*Buffer) DecodeVarint() (uint64, error)       { panic("not implemented") }
func (*Buffer) DecodeZigzag32() (uint64, error)     { panic("not implemented") }
func (*Buffer) DecodeZigzag64() (uint64, error)     { panic("not implemented") }
func (*Buffer) EncodeFixed32(uint64) error          { panic("not implemented") }
func (*Buffer) EncodeFixed64(uint64) error          { panic("not implemented") }
func (*Buffer) EncodeMessage(Message) error         { panic("not implemented") }
func (*Buffer) EncodeRawBytes([]byte) error         { panic("not implemented") }
func (*Buffer) EncodeStringBytes(string) error      { panic("not implemented") }
func (*Buffer) EncodeVarint(uint64) error           { panic("not implemented") }
func (*Buffer) EncodeZigzag32(uint64) error         { panic("not implemented") }
func (*Buffer) EncodeZigzag64(uint64) error         { panic("not implemented") }
func (*Buffer) Marshal(Message) error               { panic("not implemented") }
func (*Buffer) Unmarshal(Message) error             { panic("not implemented") }

type InternalMessageInfo struct{}

func (*InternalMessageInfo) DiscardUnknown(Message)                        { panic("not implemented") }
func (*InternalMessageInfo) Marshal([]byte, Message, bool) ([]byte, error) { panic("not implemented") }
func (*InternalMessageInfo) Merge(Message, Message)                        { panic("not implemented") }
func (*InternalMessageInfo) Size(Message) int                              { panic("not implemented") }
func (*InternalMessageInfo) Unmarshal(Message, []byte) error               { panic("not implemented") }
