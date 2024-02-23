// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal_gengo

import (
	"unicode"
	"unicode/utf8"

	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/encoding/protowire"

	"google.golang.org/protobuf/types/descriptorpb"
)

type fileInfo struct {
	*protogen.File

	allEnums      []*enumInfo
	allMessages   []*messageInfo
	allExtensions []*extensionInfo

	allEnumsByPtr         map[*enumInfo]int    // value is index into allEnums
	allMessagesByPtr      map[*messageInfo]int // value is index into allMessages
	allMessageFieldsByPtr map[*messageInfo]*structFields

	// needRawDesc specifies whether the generator should emit logic to provide
	// the legacy raw descriptor in GZIP'd form.
	// This is updated by enum and message generation logic as necessary,
	// and checked at the end of file generation.
	needRawDesc bool
}

type structFields struct {
	count      int
	unexported map[int]string
}

func (sf *structFields) append(name string) {
	if r, _ := utf8.DecodeRuneInString(name); !unicode.IsUpper(r) {
		if sf.unexported == nil {
			sf.unexported = make(map[int]string)
		}
		sf.unexported[sf.count] = name
	}
	sf.count++
}

func newFileInfo(file *protogen.File) *fileInfo {
	f := &fileInfo{File: file}

	// Collect all enums, messages, and extensions in "flattened ordering".
	// See filetype.TypeBuilder.
	var walkMessages func([]*protogen.Message, func(*protogen.Message))
	walkMessages = func(messages []*protogen.Message, f func(*protogen.Message)) {
		for _, m := range messages {
			f(m)
			walkMessages(m.Messages, f)
		}
	}
	initEnumInfos := func(enums []*protogen.Enum) {
		for _, enum := range enums {
			f.allEnums = append(f.allEnums, newEnumInfo(f, enum))
		}
	}
	initMessageInfos := func(messages []*protogen.Message) {
		for _, message := range messages {
			f.allMessages = append(f.allMessages, newMessageInfo(f, message))
		}
	}
	initExtensionInfos := func(extensions []*protogen.Extension) {
		for _, extension := range extensions {
			f.allExtensions = append(f.allExtensions, newExtensionInfo(f, extension))
		}
	}
	initEnumInfos(f.Enums)
	initMessageInfos(f.Messages)
	initExtensionInfos(f.Extensions)
	walkMessages(f.Messages, func(m *protogen.Message) {
		initEnumInfos(m.Enums)
		initMessageInfos(m.Messages)
		initExtensionInfos(m.Extensions)
	})

	// Derive a reverse mapping of enum and message pointers to their index
	// in allEnums and allMessages.
	if len(f.allEnums) > 0 {
		f.allEnumsByPtr = make(map[*enumInfo]int)
		for i, e := range f.allEnums {
			f.allEnumsByPtr[e] = i
		}
	}
	if len(f.allMessages) > 0 {
		f.allMessagesByPtr = make(map[*messageInfo]int)
		f.allMessageFieldsByPtr = make(map[*messageInfo]*structFields)
		for i, m := range f.allMessages {
			f.allMessagesByPtr[m] = i
			f.allMessageFieldsByPtr[m] = new(structFields)
		}
	}

	return f
}

type enumInfo struct {
	*protogen.Enum

	genJSONMethod    bool
	genRawDescMethod bool
}

func newEnumInfo(f *fileInfo, enum *protogen.Enum) *enumInfo {
	e := &enumInfo{Enum: enum}
	e.genJSONMethod = true
	e.genRawDescMethod = true
	return e
}

type messageInfo struct {
	*protogen.Message

	genRawDescMethod  bool
	genExtRangeMethod bool

	isTracked bool
	hasWeak   bool
}

func newMessageInfo(f *fileInfo, message *protogen.Message) *messageInfo {
	m := &messageInfo{Message: message}
	m.genRawDescMethod = true
	m.genExtRangeMethod = true
	m.isTracked = isTrackedMessage(m)
	for _, field := range m.Fields {
		m.hasWeak = m.hasWeak || field.Desc.IsWeak()
	}
	return m
}

// isTrackedMessage reports whether field tracking is enabled on the message.
func isTrackedMessage(m *messageInfo) (tracked bool) {
	const trackFieldUse_fieldNumber = 37383685

	// Decode the option from unknown fields to avoid a dependency on the
	// annotation proto from protoc-gen-go.
	b := m.Desc.Options().(*descriptorpb.MessageOptions).ProtoReflect().GetUnknown()
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		if num == trackFieldUse_fieldNumber && typ == protowire.VarintType {
			v, _ := protowire.ConsumeVarint(b)
			tracked = protowire.DecodeBool(v)
		}
		m := protowire.ConsumeFieldValue(num, typ, b)
		b = b[m:]
	}
	return tracked
}

type extensionInfo struct {
	*protogen.Extension
}

func newExtensionInfo(f *fileInfo, extension *protogen.Extension) *extensionInfo {
	x := &extensionInfo{Extension: extension}
	return x
}
