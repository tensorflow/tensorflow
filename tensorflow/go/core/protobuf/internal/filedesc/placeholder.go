// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
)

var (
	emptyNames           = new(Names)
	emptyEnumRanges      = new(EnumRanges)
	emptyFieldRanges     = new(FieldRanges)
	emptyFieldNumbers    = new(FieldNumbers)
	emptySourceLocations = new(SourceLocations)

	emptyFiles      = new(FileImports)
	emptyMessages   = new(Messages)
	emptyFields     = new(Fields)
	emptyOneofs     = new(Oneofs)
	emptyEnums      = new(Enums)
	emptyEnumValues = new(EnumValues)
	emptyExtensions = new(Extensions)
	emptyServices   = new(Services)
)

// PlaceholderFile is a placeholder, representing only the file path.
type PlaceholderFile string

func (f PlaceholderFile) ParentFile() protoreflect.FileDescriptor       { return f }
func (f PlaceholderFile) Parent() protoreflect.Descriptor               { return nil }
func (f PlaceholderFile) Index() int                                    { return 0 }
func (f PlaceholderFile) Syntax() protoreflect.Syntax                   { return 0 }
func (f PlaceholderFile) Name() protoreflect.Name                       { return "" }
func (f PlaceholderFile) FullName() protoreflect.FullName               { return "" }
func (f PlaceholderFile) IsPlaceholder() bool                           { return true }
func (f PlaceholderFile) Options() protoreflect.ProtoMessage            { return descopts.File }
func (f PlaceholderFile) Path() string                                  { return string(f) }
func (f PlaceholderFile) Package() protoreflect.FullName                { return "" }
func (f PlaceholderFile) Imports() protoreflect.FileImports             { return emptyFiles }
func (f PlaceholderFile) Messages() protoreflect.MessageDescriptors     { return emptyMessages }
func (f PlaceholderFile) Enums() protoreflect.EnumDescriptors           { return emptyEnums }
func (f PlaceholderFile) Extensions() protoreflect.ExtensionDescriptors { return emptyExtensions }
func (f PlaceholderFile) Services() protoreflect.ServiceDescriptors     { return emptyServices }
func (f PlaceholderFile) SourceLocations() protoreflect.SourceLocations { return emptySourceLocations }
func (f PlaceholderFile) ProtoType(protoreflect.FileDescriptor)         { return }
func (f PlaceholderFile) ProtoInternal(pragma.DoNotImplement)           { return }

// PlaceholderEnum is a placeholder, representing only the full name.
type PlaceholderEnum protoreflect.FullName

func (e PlaceholderEnum) ParentFile() protoreflect.FileDescriptor   { return nil }
func (e PlaceholderEnum) Parent() protoreflect.Descriptor           { return nil }
func (e PlaceholderEnum) Index() int                                { return 0 }
func (e PlaceholderEnum) Syntax() protoreflect.Syntax               { return 0 }
func (e PlaceholderEnum) Name() protoreflect.Name                   { return protoreflect.FullName(e).Name() }
func (e PlaceholderEnum) FullName() protoreflect.FullName           { return protoreflect.FullName(e) }
func (e PlaceholderEnum) IsPlaceholder() bool                       { return true }
func (e PlaceholderEnum) Options() protoreflect.ProtoMessage        { return descopts.Enum }
func (e PlaceholderEnum) Values() protoreflect.EnumValueDescriptors { return emptyEnumValues }
func (e PlaceholderEnum) ReservedNames() protoreflect.Names         { return emptyNames }
func (e PlaceholderEnum) ReservedRanges() protoreflect.EnumRanges   { return emptyEnumRanges }
func (e PlaceholderEnum) ProtoType(protoreflect.EnumDescriptor)     { return }
func (e PlaceholderEnum) ProtoInternal(pragma.DoNotImplement)       { return }

// PlaceholderEnumValue is a placeholder, representing only the full name.
type PlaceholderEnumValue protoreflect.FullName

func (e PlaceholderEnumValue) ParentFile() protoreflect.FileDescriptor    { return nil }
func (e PlaceholderEnumValue) Parent() protoreflect.Descriptor            { return nil }
func (e PlaceholderEnumValue) Index() int                                 { return 0 }
func (e PlaceholderEnumValue) Syntax() protoreflect.Syntax                { return 0 }
func (e PlaceholderEnumValue) Name() protoreflect.Name                    { return protoreflect.FullName(e).Name() }
func (e PlaceholderEnumValue) FullName() protoreflect.FullName            { return protoreflect.FullName(e) }
func (e PlaceholderEnumValue) IsPlaceholder() bool                        { return true }
func (e PlaceholderEnumValue) Options() protoreflect.ProtoMessage         { return descopts.EnumValue }
func (e PlaceholderEnumValue) Number() protoreflect.EnumNumber            { return 0 }
func (e PlaceholderEnumValue) ProtoType(protoreflect.EnumValueDescriptor) { return }
func (e PlaceholderEnumValue) ProtoInternal(pragma.DoNotImplement)        { return }

// PlaceholderMessage is a placeholder, representing only the full name.
type PlaceholderMessage protoreflect.FullName

func (m PlaceholderMessage) ParentFile() protoreflect.FileDescriptor    { return nil }
func (m PlaceholderMessage) Parent() protoreflect.Descriptor            { return nil }
func (m PlaceholderMessage) Index() int                                 { return 0 }
func (m PlaceholderMessage) Syntax() protoreflect.Syntax                { return 0 }
func (m PlaceholderMessage) Name() protoreflect.Name                    { return protoreflect.FullName(m).Name() }
func (m PlaceholderMessage) FullName() protoreflect.FullName            { return protoreflect.FullName(m) }
func (m PlaceholderMessage) IsPlaceholder() bool                        { return true }
func (m PlaceholderMessage) Options() protoreflect.ProtoMessage         { return descopts.Message }
func (m PlaceholderMessage) IsMapEntry() bool                           { return false }
func (m PlaceholderMessage) Fields() protoreflect.FieldDescriptors      { return emptyFields }
func (m PlaceholderMessage) Oneofs() protoreflect.OneofDescriptors      { return emptyOneofs }
func (m PlaceholderMessage) ReservedNames() protoreflect.Names          { return emptyNames }
func (m PlaceholderMessage) ReservedRanges() protoreflect.FieldRanges   { return emptyFieldRanges }
func (m PlaceholderMessage) RequiredNumbers() protoreflect.FieldNumbers { return emptyFieldNumbers }
func (m PlaceholderMessage) ExtensionRanges() protoreflect.FieldRanges  { return emptyFieldRanges }
func (m PlaceholderMessage) ExtensionRangeOptions(int) protoreflect.ProtoMessage {
	panic("index out of range")
}
func (m PlaceholderMessage) Messages() protoreflect.MessageDescriptors     { return emptyMessages }
func (m PlaceholderMessage) Enums() protoreflect.EnumDescriptors           { return emptyEnums }
func (m PlaceholderMessage) Extensions() protoreflect.ExtensionDescriptors { return emptyExtensions }
func (m PlaceholderMessage) ProtoType(protoreflect.MessageDescriptor)      { return }
func (m PlaceholderMessage) ProtoInternal(pragma.DoNotImplement)           { return }
