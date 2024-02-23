// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"fmt"
	"strings"

	"google.golang.org/protobuf/internal/encoding/defval"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

// ToFileDescriptorProto copies a [protoreflect.FileDescriptor] into a
// google.protobuf.FileDescriptorProto message.
func ToFileDescriptorProto(file protoreflect.FileDescriptor) *descriptorpb.FileDescriptorProto {
	p := &descriptorpb.FileDescriptorProto{
		Name:    proto.String(file.Path()),
		Options: proto.Clone(file.Options()).(*descriptorpb.FileOptions),
	}
	if file.Package() != "" {
		p.Package = proto.String(string(file.Package()))
	}
	for i, imports := 0, file.Imports(); i < imports.Len(); i++ {
		imp := imports.Get(i)
		p.Dependency = append(p.Dependency, imp.Path())
		if imp.IsPublic {
			p.PublicDependency = append(p.PublicDependency, int32(i))
		}
		if imp.IsWeak {
			p.WeakDependency = append(p.WeakDependency, int32(i))
		}
	}
	for i, locs := 0, file.SourceLocations(); i < locs.Len(); i++ {
		loc := locs.Get(i)
		l := &descriptorpb.SourceCodeInfo_Location{}
		l.Path = append(l.Path, loc.Path...)
		if loc.StartLine == loc.EndLine {
			l.Span = []int32{int32(loc.StartLine), int32(loc.StartColumn), int32(loc.EndColumn)}
		} else {
			l.Span = []int32{int32(loc.StartLine), int32(loc.StartColumn), int32(loc.EndLine), int32(loc.EndColumn)}
		}
		l.LeadingDetachedComments = append([]string(nil), loc.LeadingDetachedComments...)
		if loc.LeadingComments != "" {
			l.LeadingComments = proto.String(loc.LeadingComments)
		}
		if loc.TrailingComments != "" {
			l.TrailingComments = proto.String(loc.TrailingComments)
		}
		if p.SourceCodeInfo == nil {
			p.SourceCodeInfo = &descriptorpb.SourceCodeInfo{}
		}
		p.SourceCodeInfo.Location = append(p.SourceCodeInfo.Location, l)

	}
	for i, messages := 0, file.Messages(); i < messages.Len(); i++ {
		p.MessageType = append(p.MessageType, ToDescriptorProto(messages.Get(i)))
	}
	for i, enums := 0, file.Enums(); i < enums.Len(); i++ {
		p.EnumType = append(p.EnumType, ToEnumDescriptorProto(enums.Get(i)))
	}
	for i, services := 0, file.Services(); i < services.Len(); i++ {
		p.Service = append(p.Service, ToServiceDescriptorProto(services.Get(i)))
	}
	for i, exts := 0, file.Extensions(); i < exts.Len(); i++ {
		p.Extension = append(p.Extension, ToFieldDescriptorProto(exts.Get(i)))
	}
	if syntax := file.Syntax(); syntax != protoreflect.Proto2 && syntax.IsValid() {
		p.Syntax = proto.String(file.Syntax().String())
	}
	return p
}

// ToDescriptorProto copies a [protoreflect.MessageDescriptor] into a
// google.protobuf.DescriptorProto message.
func ToDescriptorProto(message protoreflect.MessageDescriptor) *descriptorpb.DescriptorProto {
	p := &descriptorpb.DescriptorProto{
		Name:    proto.String(string(message.Name())),
		Options: proto.Clone(message.Options()).(*descriptorpb.MessageOptions),
	}
	for i, fields := 0, message.Fields(); i < fields.Len(); i++ {
		p.Field = append(p.Field, ToFieldDescriptorProto(fields.Get(i)))
	}
	for i, exts := 0, message.Extensions(); i < exts.Len(); i++ {
		p.Extension = append(p.Extension, ToFieldDescriptorProto(exts.Get(i)))
	}
	for i, messages := 0, message.Messages(); i < messages.Len(); i++ {
		p.NestedType = append(p.NestedType, ToDescriptorProto(messages.Get(i)))
	}
	for i, enums := 0, message.Enums(); i < enums.Len(); i++ {
		p.EnumType = append(p.EnumType, ToEnumDescriptorProto(enums.Get(i)))
	}
	for i, xranges := 0, message.ExtensionRanges(); i < xranges.Len(); i++ {
		xrange := xranges.Get(i)
		p.ExtensionRange = append(p.ExtensionRange, &descriptorpb.DescriptorProto_ExtensionRange{
			Start:   proto.Int32(int32(xrange[0])),
			End:     proto.Int32(int32(xrange[1])),
			Options: proto.Clone(message.ExtensionRangeOptions(i)).(*descriptorpb.ExtensionRangeOptions),
		})
	}
	for i, oneofs := 0, message.Oneofs(); i < oneofs.Len(); i++ {
		p.OneofDecl = append(p.OneofDecl, ToOneofDescriptorProto(oneofs.Get(i)))
	}
	for i, ranges := 0, message.ReservedRanges(); i < ranges.Len(); i++ {
		rrange := ranges.Get(i)
		p.ReservedRange = append(p.ReservedRange, &descriptorpb.DescriptorProto_ReservedRange{
			Start: proto.Int32(int32(rrange[0])),
			End:   proto.Int32(int32(rrange[1])),
		})
	}
	for i, names := 0, message.ReservedNames(); i < names.Len(); i++ {
		p.ReservedName = append(p.ReservedName, string(names.Get(i)))
	}
	return p
}

// ToFieldDescriptorProto copies a [protoreflect.FieldDescriptor] into a
// google.protobuf.FieldDescriptorProto message.
func ToFieldDescriptorProto(field protoreflect.FieldDescriptor) *descriptorpb.FieldDescriptorProto {
	p := &descriptorpb.FieldDescriptorProto{
		Name:    proto.String(string(field.Name())),
		Number:  proto.Int32(int32(field.Number())),
		Label:   descriptorpb.FieldDescriptorProto_Label(field.Cardinality()).Enum(),
		Options: proto.Clone(field.Options()).(*descriptorpb.FieldOptions),
	}
	if field.IsExtension() {
		p.Extendee = fullNameOf(field.ContainingMessage())
	}
	if field.Kind().IsValid() {
		p.Type = descriptorpb.FieldDescriptorProto_Type(field.Kind()).Enum()
	}
	if field.Enum() != nil {
		p.TypeName = fullNameOf(field.Enum())
	}
	if field.Message() != nil {
		p.TypeName = fullNameOf(field.Message())
	}
	if field.HasJSONName() {
		// A bug in older versions of protoc would always populate the
		// "json_name" option for extensions when it is meaningless.
		// When it did so, it would always use the camel-cased field name.
		if field.IsExtension() {
			p.JsonName = proto.String(strs.JSONCamelCase(string(field.Name())))
		} else {
			p.JsonName = proto.String(field.JSONName())
		}
	}
	if field.Syntax() == protoreflect.Proto3 && field.HasOptionalKeyword() {
		p.Proto3Optional = proto.Bool(true)
	}
	if field.HasDefault() {
		def, err := defval.Marshal(field.Default(), field.DefaultEnumValue(), field.Kind(), defval.Descriptor)
		if err != nil && field.DefaultEnumValue() != nil {
			def = string(field.DefaultEnumValue().Name()) // occurs for unresolved enum values
		} else if err != nil {
			panic(fmt.Sprintf("%v: %v", field.FullName(), err))
		}
		p.DefaultValue = proto.String(def)
	}
	if oneof := field.ContainingOneof(); oneof != nil {
		p.OneofIndex = proto.Int32(int32(oneof.Index()))
	}
	return p
}

// ToOneofDescriptorProto copies a [protoreflect.OneofDescriptor] into a
// google.protobuf.OneofDescriptorProto message.
func ToOneofDescriptorProto(oneof protoreflect.OneofDescriptor) *descriptorpb.OneofDescriptorProto {
	return &descriptorpb.OneofDescriptorProto{
		Name:    proto.String(string(oneof.Name())),
		Options: proto.Clone(oneof.Options()).(*descriptorpb.OneofOptions),
	}
}

// ToEnumDescriptorProto copies a [protoreflect.EnumDescriptor] into a
// google.protobuf.EnumDescriptorProto message.
func ToEnumDescriptorProto(enum protoreflect.EnumDescriptor) *descriptorpb.EnumDescriptorProto {
	p := &descriptorpb.EnumDescriptorProto{
		Name:    proto.String(string(enum.Name())),
		Options: proto.Clone(enum.Options()).(*descriptorpb.EnumOptions),
	}
	for i, values := 0, enum.Values(); i < values.Len(); i++ {
		p.Value = append(p.Value, ToEnumValueDescriptorProto(values.Get(i)))
	}
	for i, ranges := 0, enum.ReservedRanges(); i < ranges.Len(); i++ {
		rrange := ranges.Get(i)
		p.ReservedRange = append(p.ReservedRange, &descriptorpb.EnumDescriptorProto_EnumReservedRange{
			Start: proto.Int32(int32(rrange[0])),
			End:   proto.Int32(int32(rrange[1])),
		})
	}
	for i, names := 0, enum.ReservedNames(); i < names.Len(); i++ {
		p.ReservedName = append(p.ReservedName, string(names.Get(i)))
	}
	return p
}

// ToEnumValueDescriptorProto copies a [protoreflect.EnumValueDescriptor] into a
// google.protobuf.EnumValueDescriptorProto message.
func ToEnumValueDescriptorProto(value protoreflect.EnumValueDescriptor) *descriptorpb.EnumValueDescriptorProto {
	return &descriptorpb.EnumValueDescriptorProto{
		Name:    proto.String(string(value.Name())),
		Number:  proto.Int32(int32(value.Number())),
		Options: proto.Clone(value.Options()).(*descriptorpb.EnumValueOptions),
	}
}

// ToServiceDescriptorProto copies a [protoreflect.ServiceDescriptor] into a
// google.protobuf.ServiceDescriptorProto message.
func ToServiceDescriptorProto(service protoreflect.ServiceDescriptor) *descriptorpb.ServiceDescriptorProto {
	p := &descriptorpb.ServiceDescriptorProto{
		Name:    proto.String(string(service.Name())),
		Options: proto.Clone(service.Options()).(*descriptorpb.ServiceOptions),
	}
	for i, methods := 0, service.Methods(); i < methods.Len(); i++ {
		p.Method = append(p.Method, ToMethodDescriptorProto(methods.Get(i)))
	}
	return p
}

// ToMethodDescriptorProto copies a [protoreflect.MethodDescriptor] into a
// google.protobuf.MethodDescriptorProto message.
func ToMethodDescriptorProto(method protoreflect.MethodDescriptor) *descriptorpb.MethodDescriptorProto {
	p := &descriptorpb.MethodDescriptorProto{
		Name:       proto.String(string(method.Name())),
		InputType:  fullNameOf(method.Input()),
		OutputType: fullNameOf(method.Output()),
		Options:    proto.Clone(method.Options()).(*descriptorpb.MethodOptions),
	}
	if method.IsStreamingClient() {
		p.ClientStreaming = proto.Bool(true)
	}
	if method.IsStreamingServer() {
		p.ServerStreaming = proto.Bool(true)
	}
	return p
}

func fullNameOf(d protoreflect.Descriptor) *string {
	if d == nil {
		return nil
	}
	if strings.HasPrefix(string(d.FullName()), unknownPrefix) {
		return proto.String(string(d.FullName()[len(unknownPrefix):]))
	}
	return proto.String("." + string(d.FullName()))
}
