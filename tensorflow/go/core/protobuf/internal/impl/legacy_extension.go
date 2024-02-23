// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"reflect"

	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/encoding/messageset"
	ptag "google.golang.org/protobuf/internal/encoding/tag"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
)

func (xi *ExtensionInfo) initToLegacy() {
	xd := xi.desc
	var parent protoiface.MessageV1
	messageName := xd.ContainingMessage().FullName()
	if mt, _ := protoregistry.GlobalTypes.FindMessageByName(messageName); mt != nil {
		// Create a new parent message and unwrap it if possible.
		mv := mt.New().Interface()
		t := reflect.TypeOf(mv)
		if mv, ok := mv.(unwrapper); ok {
			t = reflect.TypeOf(mv.protoUnwrap())
		}

		// Check whether the message implements the legacy v1 Message interface.
		mz := reflect.Zero(t).Interface()
		if mz, ok := mz.(protoiface.MessageV1); ok {
			parent = mz
		}
	}

	// Determine the v1 extension type, which is unfortunately not the same as
	// the v2 ExtensionType.GoType.
	extType := xi.goType
	switch extType.Kind() {
	case reflect.Bool, reflect.Int32, reflect.Int64, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64, reflect.String:
		extType = reflect.PtrTo(extType) // T -> *T for singular scalar fields
	}

	// Reconstruct the legacy enum full name.
	var enumName string
	if xd.Kind() == protoreflect.EnumKind {
		enumName = legacyEnumName(xd.Enum())
	}

	// Derive the proto file that the extension was declared within.
	var filename string
	if fd := xd.ParentFile(); fd != nil {
		filename = fd.Path()
	}

	// For MessageSet extensions, the name used is the parent message.
	name := xd.FullName()
	if messageset.IsMessageSetExtension(xd) {
		name = name.Parent()
	}

	xi.ExtendedType = parent
	xi.ExtensionType = reflect.Zero(extType).Interface()
	xi.Field = int32(xd.Number())
	xi.Name = string(name)
	xi.Tag = ptag.Marshal(xd, enumName)
	xi.Filename = filename
}

// initFromLegacy initializes an ExtensionInfo from
// the contents of the deprecated exported fields of the type.
func (xi *ExtensionInfo) initFromLegacy() {
	// The v1 API returns "type incomplete" descriptors where only the
	// field number is specified. In such a case, use a placeholder.
	if xi.ExtendedType == nil || xi.ExtensionType == nil {
		xd := placeholderExtension{
			name:   protoreflect.FullName(xi.Name),
			number: protoreflect.FieldNumber(xi.Field),
		}
		xi.desc = extensionTypeDescriptor{xd, xi}
		return
	}

	// Resolve enum or message dependencies.
	var ed protoreflect.EnumDescriptor
	var md protoreflect.MessageDescriptor
	t := reflect.TypeOf(xi.ExtensionType)
	isOptional := t.Kind() == reflect.Ptr && t.Elem().Kind() != reflect.Struct
	isRepeated := t.Kind() == reflect.Slice && t.Elem().Kind() != reflect.Uint8
	if isOptional || isRepeated {
		t = t.Elem()
	}
	switch v := reflect.Zero(t).Interface().(type) {
	case protoreflect.Enum:
		ed = v.Descriptor()
	case enumV1:
		ed = LegacyLoadEnumDesc(t)
	case protoreflect.ProtoMessage:
		md = v.ProtoReflect().Descriptor()
	case messageV1:
		md = LegacyLoadMessageDesc(t)
	}

	// Derive basic field information from the struct tag.
	var evs protoreflect.EnumValueDescriptors
	if ed != nil {
		evs = ed.Values()
	}
	fd := ptag.Unmarshal(xi.Tag, t, evs).(*filedesc.Field)

	// Construct a v2 ExtensionType.
	xd := &filedesc.Extension{L2: new(filedesc.ExtensionL2)}
	xd.L0.ParentFile = filedesc.SurrogateProto2
	xd.L0.FullName = protoreflect.FullName(xi.Name)
	xd.L1.Number = protoreflect.FieldNumber(xi.Field)
	xd.L1.Cardinality = fd.L1.Cardinality
	xd.L1.Kind = fd.L1.Kind
	xd.L2.IsPacked = fd.L1.IsPacked
	xd.L2.Default = fd.L1.Default
	xd.L1.Extendee = Export{}.MessageDescriptorOf(xi.ExtendedType)
	xd.L2.Enum = ed
	xd.L2.Message = md

	// Derive real extension field name for MessageSets.
	if messageset.IsMessageSet(xd.L1.Extendee) && md.FullName() == xd.L0.FullName {
		xd.L0.FullName = xd.L0.FullName.Append(messageset.ExtensionName)
	}

	tt := reflect.TypeOf(xi.ExtensionType)
	if isOptional {
		tt = tt.Elem()
	}
	xi.goType = tt
	xi.desc = extensionTypeDescriptor{xd, xi}
}

type placeholderExtension struct {
	name   protoreflect.FullName
	number protoreflect.FieldNumber
}

func (x placeholderExtension) ParentFile() protoreflect.FileDescriptor            { return nil }
func (x placeholderExtension) Parent() protoreflect.Descriptor                    { return nil }
func (x placeholderExtension) Index() int                                         { return 0 }
func (x placeholderExtension) Syntax() protoreflect.Syntax                        { return 0 }
func (x placeholderExtension) Name() protoreflect.Name                            { return x.name.Name() }
func (x placeholderExtension) FullName() protoreflect.FullName                    { return x.name }
func (x placeholderExtension) IsPlaceholder() bool                                { return true }
func (x placeholderExtension) Options() protoreflect.ProtoMessage                 { return descopts.Field }
func (x placeholderExtension) Number() protoreflect.FieldNumber                   { return x.number }
func (x placeholderExtension) Cardinality() protoreflect.Cardinality              { return 0 }
func (x placeholderExtension) Kind() protoreflect.Kind                            { return 0 }
func (x placeholderExtension) HasJSONName() bool                                  { return false }
func (x placeholderExtension) JSONName() string                                   { return "[" + string(x.name) + "]" }
func (x placeholderExtension) TextName() string                                   { return "[" + string(x.name) + "]" }
func (x placeholderExtension) HasPresence() bool                                  { return false }
func (x placeholderExtension) HasOptionalKeyword() bool                           { return false }
func (x placeholderExtension) IsExtension() bool                                  { return true }
func (x placeholderExtension) IsWeak() bool                                       { return false }
func (x placeholderExtension) IsPacked() bool                                     { return false }
func (x placeholderExtension) IsList() bool                                       { return false }
func (x placeholderExtension) IsMap() bool                                        { return false }
func (x placeholderExtension) MapKey() protoreflect.FieldDescriptor               { return nil }
func (x placeholderExtension) MapValue() protoreflect.FieldDescriptor             { return nil }
func (x placeholderExtension) HasDefault() bool                                   { return false }
func (x placeholderExtension) Default() protoreflect.Value                        { return protoreflect.Value{} }
func (x placeholderExtension) DefaultEnumValue() protoreflect.EnumValueDescriptor { return nil }
func (x placeholderExtension) ContainingOneof() protoreflect.OneofDescriptor      { return nil }
func (x placeholderExtension) ContainingMessage() protoreflect.MessageDescriptor  { return nil }
func (x placeholderExtension) Enum() protoreflect.EnumDescriptor                  { return nil }
func (x placeholderExtension) Message() protoreflect.MessageDescriptor            { return nil }
func (x placeholderExtension) ProtoType(protoreflect.FieldDescriptor)             { return }
func (x placeholderExtension) ProtoInternal(pragma.DoNotImplement)                { return }
