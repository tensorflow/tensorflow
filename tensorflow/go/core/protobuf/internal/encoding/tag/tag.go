// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tag marshals and unmarshals the legacy struct tags as generated
// by historical versions of protoc-gen-go.
package tag

import (
	"reflect"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/encoding/defval"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/reflect/protoreflect"
)

var byteType = reflect.TypeOf(byte(0))

// Unmarshal decodes the tag into a prototype.Field.
//
// The goType is needed to determine the original protoreflect.Kind since the
// tag does not record sufficient information to determine that.
// The type is the underlying field type (e.g., a repeated field may be
// represented by []T, but the Go type passed in is just T).
// A list of enum value descriptors must be provided for enum fields.
// This does not populate the Enum or Message (except for weak message).
//
// This function is a best effort attempt; parsing errors are ignored.
func Unmarshal(tag string, goType reflect.Type, evs protoreflect.EnumValueDescriptors) protoreflect.FieldDescriptor {
	f := new(filedesc.Field)
	f.L0.ParentFile = filedesc.SurrogateProto2
	for len(tag) > 0 {
		i := strings.IndexByte(tag, ',')
		if i < 0 {
			i = len(tag)
		}
		switch s := tag[:i]; {
		case strings.HasPrefix(s, "name="):
			f.L0.FullName = protoreflect.FullName(s[len("name="):])
		case strings.Trim(s, "0123456789") == "":
			n, _ := strconv.ParseUint(s, 10, 32)
			f.L1.Number = protoreflect.FieldNumber(n)
		case s == "opt":
			f.L1.Cardinality = protoreflect.Optional
		case s == "req":
			f.L1.Cardinality = protoreflect.Required
		case s == "rep":
			f.L1.Cardinality = protoreflect.Repeated
		case s == "varint":
			switch goType.Kind() {
			case reflect.Bool:
				f.L1.Kind = protoreflect.BoolKind
			case reflect.Int32:
				f.L1.Kind = protoreflect.Int32Kind
			case reflect.Int64:
				f.L1.Kind = protoreflect.Int64Kind
			case reflect.Uint32:
				f.L1.Kind = protoreflect.Uint32Kind
			case reflect.Uint64:
				f.L1.Kind = protoreflect.Uint64Kind
			}
		case s == "zigzag32":
			if goType.Kind() == reflect.Int32 {
				f.L1.Kind = protoreflect.Sint32Kind
			}
		case s == "zigzag64":
			if goType.Kind() == reflect.Int64 {
				f.L1.Kind = protoreflect.Sint64Kind
			}
		case s == "fixed32":
			switch goType.Kind() {
			case reflect.Int32:
				f.L1.Kind = protoreflect.Sfixed32Kind
			case reflect.Uint32:
				f.L1.Kind = protoreflect.Fixed32Kind
			case reflect.Float32:
				f.L1.Kind = protoreflect.FloatKind
			}
		case s == "fixed64":
			switch goType.Kind() {
			case reflect.Int64:
				f.L1.Kind = protoreflect.Sfixed64Kind
			case reflect.Uint64:
				f.L1.Kind = protoreflect.Fixed64Kind
			case reflect.Float64:
				f.L1.Kind = protoreflect.DoubleKind
			}
		case s == "bytes":
			switch {
			case goType.Kind() == reflect.String:
				f.L1.Kind = protoreflect.StringKind
			case goType.Kind() == reflect.Slice && goType.Elem() == byteType:
				f.L1.Kind = protoreflect.BytesKind
			default:
				f.L1.Kind = protoreflect.MessageKind
			}
		case s == "group":
			f.L1.Kind = protoreflect.GroupKind
		case strings.HasPrefix(s, "enum="):
			f.L1.Kind = protoreflect.EnumKind
		case strings.HasPrefix(s, "json="):
			jsonName := s[len("json="):]
			if jsonName != strs.JSONCamelCase(string(f.L0.FullName.Name())) {
				f.L1.StringName.InitJSON(jsonName)
			}
		case s == "packed":
			f.L1.HasPacked = true
			f.L1.IsPacked = true
		case strings.HasPrefix(s, "weak="):
			f.L1.IsWeak = true
			f.L1.Message = filedesc.PlaceholderMessage(protoreflect.FullName(s[len("weak="):]))
		case strings.HasPrefix(s, "def="):
			// The default tag is special in that everything afterwards is the
			// default regardless of the presence of commas.
			s, i = tag[len("def="):], len(tag)
			v, ev, _ := defval.Unmarshal(s, f.L1.Kind, evs, defval.GoTag)
			f.L1.Default = filedesc.DefaultValue(v, ev)
		case s == "proto3":
			f.L0.ParentFile = filedesc.SurrogateProto3
		}
		tag = strings.TrimPrefix(tag[i:], ",")
	}

	// The generator uses the group message name instead of the field name.
	// We obtain the real field name by lowercasing the group name.
	if f.L1.Kind == protoreflect.GroupKind {
		f.L0.FullName = protoreflect.FullName(strings.ToLower(string(f.L0.FullName)))
	}
	return f
}

// Marshal encodes the protoreflect.FieldDescriptor as a tag.
//
// The enumName must be provided if the kind is an enum.
// Historically, the formulation of the enum "name" was the proto package
// dot-concatenated with the generated Go identifier for the enum type.
// Depending on the context on how Marshal is called, there are different ways
// through which that information is determined. As such it is the caller's
// responsibility to provide a function to obtain that information.
func Marshal(fd protoreflect.FieldDescriptor, enumName string) string {
	var tag []string
	switch fd.Kind() {
	case protoreflect.BoolKind, protoreflect.EnumKind, protoreflect.Int32Kind, protoreflect.Uint32Kind, protoreflect.Int64Kind, protoreflect.Uint64Kind:
		tag = append(tag, "varint")
	case protoreflect.Sint32Kind:
		tag = append(tag, "zigzag32")
	case protoreflect.Sint64Kind:
		tag = append(tag, "zigzag64")
	case protoreflect.Sfixed32Kind, protoreflect.Fixed32Kind, protoreflect.FloatKind:
		tag = append(tag, "fixed32")
	case protoreflect.Sfixed64Kind, protoreflect.Fixed64Kind, protoreflect.DoubleKind:
		tag = append(tag, "fixed64")
	case protoreflect.StringKind, protoreflect.BytesKind, protoreflect.MessageKind:
		tag = append(tag, "bytes")
	case protoreflect.GroupKind:
		tag = append(tag, "group")
	}
	tag = append(tag, strconv.Itoa(int(fd.Number())))
	switch fd.Cardinality() {
	case protoreflect.Optional:
		tag = append(tag, "opt")
	case protoreflect.Required:
		tag = append(tag, "req")
	case protoreflect.Repeated:
		tag = append(tag, "rep")
	}
	if fd.IsPacked() {
		tag = append(tag, "packed")
	}
	name := string(fd.Name())
	if fd.Kind() == protoreflect.GroupKind {
		// The name of the FieldDescriptor for a group field is
		// lowercased. To find the original capitalization, we
		// look in the field's MessageType.
		name = string(fd.Message().Name())
	}
	tag = append(tag, "name="+name)
	if jsonName := fd.JSONName(); jsonName != "" && jsonName != name && !fd.IsExtension() {
		// NOTE: The jsonName != name condition is suspect, but it preserve
		// the exact same semantics from the previous generator.
		tag = append(tag, "json="+jsonName)
	}
	if fd.IsWeak() {
		tag = append(tag, "weak="+string(fd.Message().FullName()))
	}
	// The previous implementation does not tag extension fields as proto3,
	// even when the field is defined in a proto3 file. Match that behavior
	// for consistency.
	if fd.Syntax() == protoreflect.Proto3 && !fd.IsExtension() {
		tag = append(tag, "proto3")
	}
	if fd.Kind() == protoreflect.EnumKind && enumName != "" {
		tag = append(tag, "enum="+enumName)
	}
	if fd.ContainingOneof() != nil {
		tag = append(tag, "oneof")
	}
	// This must appear last in the tag, since commas in strings aren't escaped.
	if fd.HasDefault() {
		def, _ := defval.Marshal(fd.Default(), fd.DefaultEnumValue(), fd.Kind(), defval.GoTag)
		tag = append(tag, "def="+def)
	}
	return strings.Join(tag, ",")
}
