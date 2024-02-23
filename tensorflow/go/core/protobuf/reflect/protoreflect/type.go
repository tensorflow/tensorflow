// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

// Descriptor provides a set of accessors that are common to every descriptor.
// Each descriptor type wraps the equivalent google.protobuf.XXXDescriptorProto,
// but provides efficient lookup and immutability.
//
// Each descriptor is comparable. Equality implies that the two types are
// exactly identical. However, it is possible for the same semantically
// identical proto type to be represented by multiple type descriptors.
//
// For example, suppose we have t1 and t2 which are both an [MessageDescriptor].
// If t1 == t2, then the types are definitely equal and all accessors return
// the same information. However, if t1 != t2, then it is still possible that
// they still represent the same proto type (e.g., t1.FullName == t2.FullName).
// This can occur if a descriptor type is created dynamically, or multiple
// versions of the same proto type are accidentally linked into the Go binary.
type Descriptor interface {
	// ParentFile returns the parent file descriptor that this descriptor
	// is declared within. The parent file for the file descriptor is itself.
	//
	// Support for this functionality is optional and may return nil.
	ParentFile() FileDescriptor

	// Parent returns the parent containing this descriptor declaration.
	// The following shows the mapping from child type to possible parent types:
	//
	//	╔═════════════════════╤═══════════════════════════════════╗
	//	║ Child type          │ Possible parent types             ║
	//	╠═════════════════════╪═══════════════════════════════════╣
	//	║ FileDescriptor      │ nil                               ║
	//	║ MessageDescriptor   │ FileDescriptor, MessageDescriptor ║
	//	║ FieldDescriptor     │ FileDescriptor, MessageDescriptor ║
	//	║ OneofDescriptor     │ MessageDescriptor                 ║
	//	║ EnumDescriptor      │ FileDescriptor, MessageDescriptor ║
	//	║ EnumValueDescriptor │ EnumDescriptor                    ║
	//	║ ServiceDescriptor   │ FileDescriptor                    ║
	//	║ MethodDescriptor    │ ServiceDescriptor                 ║
	//	╚═════════════════════╧═══════════════════════════════════╝
	//
	// Support for this functionality is optional and may return nil.
	Parent() Descriptor

	// Index returns the index of this descriptor within its parent.
	// It returns 0 if the descriptor does not have a parent or if the parent
	// is unknown.
	Index() int

	// Syntax is the protobuf syntax.
	Syntax() Syntax // e.g., Proto2 or Proto3

	// Name is the short name of the declaration (i.e., FullName.Name).
	Name() Name // e.g., "Any"

	// FullName is the fully-qualified name of the declaration.
	//
	// The FullName is a concatenation of the full name of the type that this
	// type is declared within and the declaration name. For example,
	// field "foo_field" in message "proto.package.MyMessage" is
	// uniquely identified as "proto.package.MyMessage.foo_field".
	// Enum values are an exception to the rule (see EnumValueDescriptor).
	FullName() FullName // e.g., "google.protobuf.Any"

	// IsPlaceholder reports whether type information is missing since a
	// dependency is not resolved, in which case only name information is known.
	//
	// Placeholder types may only be returned by the following accessors
	// as a result of unresolved dependencies or weak imports:
	//
	//	╔═══════════════════════════════════╤═════════════════════╗
	//	║ Accessor                          │ Descriptor          ║
	//	╠═══════════════════════════════════╪═════════════════════╣
	//	║ FileImports.FileDescriptor        │ FileDescriptor      ║
	//	║ FieldDescriptor.Enum              │ EnumDescriptor      ║
	//	║ FieldDescriptor.Message           │ MessageDescriptor   ║
	//	║ FieldDescriptor.DefaultEnumValue  │ EnumValueDescriptor ║
	//	║ FieldDescriptor.ContainingMessage │ MessageDescriptor   ║
	//	║ MethodDescriptor.Input            │ MessageDescriptor   ║
	//	║ MethodDescriptor.Output           │ MessageDescriptor   ║
	//	╚═══════════════════════════════════╧═════════════════════╝
	//
	// If true, only Name and FullName are valid.
	// For FileDescriptor, the Path is also valid.
	IsPlaceholder() bool

	// Options returns the descriptor options. The caller must not modify
	// the returned value.
	//
	// To avoid a dependency cycle, this function returns a proto.Message value.
	// The proto message type returned for each descriptor type is as follows:
	//	╔═════════════════════╤══════════════════════════════════════════╗
	//	║ Go type             │ Protobuf message type                    ║
	//	╠═════════════════════╪══════════════════════════════════════════╣
	//	║ FileDescriptor      │ google.protobuf.FileOptions              ║
	//	║ EnumDescriptor      │ google.protobuf.EnumOptions              ║
	//	║ EnumValueDescriptor │ google.protobuf.EnumValueOptions         ║
	//	║ MessageDescriptor   │ google.protobuf.MessageOptions           ║
	//	║ FieldDescriptor     │ google.protobuf.FieldOptions             ║
	//	║ OneofDescriptor     │ google.protobuf.OneofOptions             ║
	//	║ ServiceDescriptor   │ google.protobuf.ServiceOptions           ║
	//	║ MethodDescriptor    │ google.protobuf.MethodOptions            ║
	//	╚═════════════════════╧══════════════════════════════════════════╝
	//
	// This method returns a typed nil-pointer if no options are present.
	// The caller must import the descriptorpb package to use this.
	Options() ProtoMessage

	doNotImplement
}

// FileDescriptor describes the types in a complete proto file and
// corresponds with the google.protobuf.FileDescriptorProto message.
//
// Top-level declarations:
// [EnumDescriptor], [MessageDescriptor], [FieldDescriptor], and/or [ServiceDescriptor].
type FileDescriptor interface {
	Descriptor // Descriptor.FullName is identical to Package

	// Path returns the file name, relative to the source tree root.
	Path() string // e.g., "path/to/file.proto"
	// Package returns the protobuf package namespace.
	Package() FullName // e.g., "google.protobuf"

	// Imports is a list of imported proto files.
	Imports() FileImports

	// Enums is a list of the top-level enum declarations.
	Enums() EnumDescriptors
	// Messages is a list of the top-level message declarations.
	Messages() MessageDescriptors
	// Extensions is a list of the top-level extension declarations.
	Extensions() ExtensionDescriptors
	// Services is a list of the top-level service declarations.
	Services() ServiceDescriptors

	// SourceLocations is a list of source locations.
	SourceLocations() SourceLocations

	isFileDescriptor
}
type isFileDescriptor interface{ ProtoType(FileDescriptor) }

// FileImports is a list of file imports.
type FileImports interface {
	// Len reports the number of files imported by this proto file.
	Len() int
	// Get returns the ith FileImport. It panics if out of bounds.
	Get(i int) FileImport

	doNotImplement
}

// FileImport is the declaration for a proto file import.
type FileImport struct {
	// FileDescriptor is the file type for the given import.
	// It is a placeholder descriptor if IsWeak is set or if a dependency has
	// not been regenerated to implement the new reflection APIs.
	FileDescriptor

	// IsPublic reports whether this is a public import, which causes this file
	// to alias declarations within the imported file. The intended use cases
	// for this feature is the ability to move proto files without breaking
	// existing dependencies.
	//
	// The current file and the imported file must be within proto package.
	IsPublic bool

	// IsWeak reports whether this is a weak import, which does not impose
	// a direct dependency on the target file.
	//
	// Weak imports are a legacy proto1 feature. Equivalent behavior is
	// achieved using proto2 extension fields or proto3 Any messages.
	IsWeak bool
}

// MessageDescriptor describes a message and
// corresponds with the google.protobuf.DescriptorProto message.
//
// Nested declarations:
// [FieldDescriptor], [OneofDescriptor], [FieldDescriptor], [EnumDescriptor],
// and/or [MessageDescriptor].
type MessageDescriptor interface {
	Descriptor

	// IsMapEntry indicates that this is an auto-generated message type to
	// represent the entry type for a map field.
	//
	// Map entry messages have only two fields:
	//	• a "key" field with a field number of 1
	//	• a "value" field with a field number of 2
	// The key and value types are determined by these two fields.
	//
	// If IsMapEntry is true, it implies that FieldDescriptor.IsMap is true
	// for some field with this message type.
	IsMapEntry() bool

	// Fields is a list of nested field declarations.
	Fields() FieldDescriptors
	// Oneofs is a list of nested oneof declarations.
	Oneofs() OneofDescriptors

	// ReservedNames is a list of reserved field names.
	ReservedNames() Names
	// ReservedRanges is a list of reserved ranges of field numbers.
	ReservedRanges() FieldRanges
	// RequiredNumbers is a list of required field numbers.
	// In Proto3, it is always an empty list.
	RequiredNumbers() FieldNumbers
	// ExtensionRanges is the field ranges used for extension fields.
	// In Proto3, it is always an empty ranges.
	ExtensionRanges() FieldRanges
	// ExtensionRangeOptions returns the ith extension range options.
	//
	// To avoid a dependency cycle, this method returns a proto.Message] value,
	// which always contains a google.protobuf.ExtensionRangeOptions message.
	// This method returns a typed nil-pointer if no options are present.
	// The caller must import the descriptorpb package to use this.
	ExtensionRangeOptions(i int) ProtoMessage

	// Enums is a list of nested enum declarations.
	Enums() EnumDescriptors
	// Messages is a list of nested message declarations.
	Messages() MessageDescriptors
	// Extensions is a list of nested extension declarations.
	Extensions() ExtensionDescriptors

	isMessageDescriptor
}
type isMessageDescriptor interface{ ProtoType(MessageDescriptor) }

// MessageType encapsulates a [MessageDescriptor] with a concrete Go implementation.
// It is recommended that implementations of this interface also implement the
// [MessageFieldTypes] interface.
type MessageType interface {
	// New returns a newly allocated empty message.
	// It may return nil for synthetic messages representing a map entry.
	New() Message

	// Zero returns an empty, read-only message.
	// It may return nil for synthetic messages representing a map entry.
	Zero() Message

	// Descriptor returns the message descriptor.
	//
	// Invariant: t.Descriptor() == t.New().Descriptor()
	Descriptor() MessageDescriptor
}

// MessageFieldTypes extends a [MessageType] by providing type information
// regarding enums and messages referenced by the message fields.
type MessageFieldTypes interface {
	MessageType

	// Enum returns the EnumType for the ith field in MessageDescriptor.Fields.
	// It returns nil if the ith field is not an enum kind.
	// It panics if out of bounds.
	//
	// Invariant: mt.Enum(i).Descriptor() == mt.Descriptor().Fields(i).Enum()
	Enum(i int) EnumType

	// Message returns the MessageType for the ith field in MessageDescriptor.Fields.
	// It returns nil if the ith field is not a message or group kind.
	// It panics if out of bounds.
	//
	// Invariant: mt.Message(i).Descriptor() == mt.Descriptor().Fields(i).Message()
	Message(i int) MessageType
}

// MessageDescriptors is a list of message declarations.
type MessageDescriptors interface {
	// Len reports the number of messages.
	Len() int
	// Get returns the ith MessageDescriptor. It panics if out of bounds.
	Get(i int) MessageDescriptor
	// ByName returns the MessageDescriptor for a message named s.
	// It returns nil if not found.
	ByName(s Name) MessageDescriptor

	doNotImplement
}

// FieldDescriptor describes a field within a message and
// corresponds with the google.protobuf.FieldDescriptorProto message.
//
// It is used for both normal fields defined within the parent message
// (e.g., [MessageDescriptor.Fields]) and fields that extend some remote message
// (e.g., [FileDescriptor.Extensions] or [MessageDescriptor.Extensions]).
type FieldDescriptor interface {
	Descriptor

	// Number reports the unique number for this field.
	Number() FieldNumber
	// Cardinality reports the cardinality for this field.
	Cardinality() Cardinality
	// Kind reports the basic kind for this field.
	Kind() Kind

	// HasJSONName reports whether this field has an explicitly set JSON name.
	HasJSONName() bool

	// JSONName reports the name used for JSON serialization.
	// It is usually the camel-cased form of the field name.
	// Extension fields are represented by the full name surrounded by brackets.
	JSONName() string

	// TextName reports the name used for text serialization.
	// It is usually the name of the field, except that groups use the name
	// of the inlined message, and extension fields are represented by the
	// full name surrounded by brackets.
	TextName() string

	// HasPresence reports whether the field distinguishes between unpopulated
	// and default values.
	HasPresence() bool

	// IsExtension reports whether this is an extension field. If false,
	// then Parent and ContainingMessage refer to the same message.
	// Otherwise, ContainingMessage and Parent likely differ.
	IsExtension() bool

	// HasOptionalKeyword reports whether the "optional" keyword was explicitly
	// specified in the source .proto file.
	HasOptionalKeyword() bool

	// IsWeak reports whether this is a weak field, which does not impose a
	// direct dependency on the target type.
	// If true, then Message returns a placeholder type.
	IsWeak() bool

	// IsPacked reports whether repeated primitive numeric kinds should be
	// serialized using a packed encoding.
	// If true, then it implies Cardinality is Repeated.
	IsPacked() bool

	// IsList reports whether this field represents a list,
	// where the value type for the associated field is a List.
	// It is equivalent to checking whether Cardinality is Repeated and
	// that IsMap reports false.
	IsList() bool

	// IsMap reports whether this field represents a map,
	// where the value type for the associated field is a Map.
	// It is equivalent to checking whether Cardinality is Repeated,
	// that the Kind is MessageKind, and that MessageDescriptor.IsMapEntry reports true.
	IsMap() bool

	// MapKey returns the field descriptor for the key in the map entry.
	// It returns nil if IsMap reports false.
	MapKey() FieldDescriptor

	// MapValue returns the field descriptor for the value in the map entry.
	// It returns nil if IsMap reports false.
	MapValue() FieldDescriptor

	// HasDefault reports whether this field has a default value.
	HasDefault() bool

	// Default returns the default value for scalar fields.
	// For proto2, it is the default value as specified in the proto file,
	// or the zero value if unspecified.
	// For proto3, it is always the zero value of the scalar.
	// The Value type is determined by the Kind.
	Default() Value

	// DefaultEnumValue returns the enum value descriptor for the default value
	// of an enum field, and is nil for any other kind of field.
	DefaultEnumValue() EnumValueDescriptor

	// ContainingOneof is the containing oneof that this field belongs to,
	// and is nil if this field is not part of a oneof.
	ContainingOneof() OneofDescriptor

	// ContainingMessage is the containing message that this field belongs to.
	// For extension fields, this may not necessarily be the parent message
	// that the field is declared within.
	ContainingMessage() MessageDescriptor

	// Enum is the enum descriptor if Kind is EnumKind.
	// It returns nil for any other Kind.
	Enum() EnumDescriptor

	// Message is the message descriptor if Kind is
	// MessageKind or GroupKind. It returns nil for any other Kind.
	Message() MessageDescriptor

	isFieldDescriptor
}
type isFieldDescriptor interface{ ProtoType(FieldDescriptor) }

// FieldDescriptors is a list of field declarations.
type FieldDescriptors interface {
	// Len reports the number of fields.
	Len() int
	// Get returns the ith FieldDescriptor. It panics if out of bounds.
	Get(i int) FieldDescriptor
	// ByName returns the FieldDescriptor for a field named s.
	// It returns nil if not found.
	ByName(s Name) FieldDescriptor
	// ByJSONName returns the FieldDescriptor for a field with s as the JSON name.
	// It returns nil if not found.
	ByJSONName(s string) FieldDescriptor
	// ByTextName returns the FieldDescriptor for a field with s as the text name.
	// It returns nil if not found.
	ByTextName(s string) FieldDescriptor
	// ByNumber returns the FieldDescriptor for a field numbered n.
	// It returns nil if not found.
	ByNumber(n FieldNumber) FieldDescriptor

	doNotImplement
}

// OneofDescriptor describes a oneof field set within a given message and
// corresponds with the google.protobuf.OneofDescriptorProto message.
type OneofDescriptor interface {
	Descriptor

	// IsSynthetic reports whether this is a synthetic oneof created to support
	// proto3 optional semantics. If true, Fields contains exactly one field
	// with FieldDescriptor.HasOptionalKeyword specified.
	IsSynthetic() bool

	// Fields is a list of fields belonging to this oneof.
	Fields() FieldDescriptors

	isOneofDescriptor
}
type isOneofDescriptor interface{ ProtoType(OneofDescriptor) }

// OneofDescriptors is a list of oneof declarations.
type OneofDescriptors interface {
	// Len reports the number of oneof fields.
	Len() int
	// Get returns the ith OneofDescriptor. It panics if out of bounds.
	Get(i int) OneofDescriptor
	// ByName returns the OneofDescriptor for a oneof named s.
	// It returns nil if not found.
	ByName(s Name) OneofDescriptor

	doNotImplement
}

// ExtensionDescriptor is an alias of [FieldDescriptor] for documentation.
type ExtensionDescriptor = FieldDescriptor

// ExtensionTypeDescriptor is an [ExtensionDescriptor] with an associated [ExtensionType].
type ExtensionTypeDescriptor interface {
	ExtensionDescriptor

	// Type returns the associated ExtensionType.
	Type() ExtensionType

	// Descriptor returns the plain ExtensionDescriptor without the
	// associated ExtensionType.
	Descriptor() ExtensionDescriptor
}

// ExtensionDescriptors is a list of field declarations.
type ExtensionDescriptors interface {
	// Len reports the number of fields.
	Len() int
	// Get returns the ith ExtensionDescriptor. It panics if out of bounds.
	Get(i int) ExtensionDescriptor
	// ByName returns the ExtensionDescriptor for a field named s.
	// It returns nil if not found.
	ByName(s Name) ExtensionDescriptor

	doNotImplement
}

// ExtensionType encapsulates an [ExtensionDescriptor] with a concrete
// Go implementation. The nested field descriptor must be for a extension field.
//
// While a normal field is a member of the parent message that it is declared
// within (see [Descriptor.Parent]), an extension field is a member of some other
// target message (see [FieldDescriptor.ContainingMessage]) and may have no
// relationship with the parent. However, the full name of an extension field is
// relative to the parent that it is declared within.
//
// For example:
//
//	syntax = "proto2";
//	package example;
//	message FooMessage {
//		extensions 100 to max;
//	}
//	message BarMessage {
//		extends FooMessage { optional BarMessage bar_field = 100; }
//	}
//
// Field "bar_field" is an extension of FooMessage, but its full name is
// "example.BarMessage.bar_field" instead of "example.FooMessage.bar_field".
type ExtensionType interface {
	// New returns a new value for the field.
	// For scalars, this returns the default value in native Go form.
	New() Value

	// Zero returns a new value for the field.
	// For scalars, this returns the default value in native Go form.
	// For composite types, this returns an empty, read-only message, list, or map.
	Zero() Value

	// TypeDescriptor returns the extension type descriptor.
	TypeDescriptor() ExtensionTypeDescriptor

	// ValueOf wraps the input and returns it as a Value.
	// ValueOf panics if the input value is invalid or not the appropriate type.
	//
	// ValueOf is more extensive than protoreflect.ValueOf for a given field's
	// value as it has more type information available.
	ValueOf(interface{}) Value

	// InterfaceOf completely unwraps the Value to the underlying Go type.
	// InterfaceOf panics if the input is nil or does not represent the
	// appropriate underlying Go type. For composite types, it panics if the
	// value is not mutable.
	//
	// InterfaceOf is able to unwrap the Value further than Value.Interface
	// as it has more type information available.
	InterfaceOf(Value) interface{}

	// IsValidValue reports whether the Value is valid to assign to the field.
	IsValidValue(Value) bool

	// IsValidInterface reports whether the input is valid to assign to the field.
	IsValidInterface(interface{}) bool
}

// EnumDescriptor describes an enum and
// corresponds with the google.protobuf.EnumDescriptorProto message.
//
// Nested declarations:
// [EnumValueDescriptor].
type EnumDescriptor interface {
	Descriptor

	// Values is a list of nested enum value declarations.
	Values() EnumValueDescriptors

	// ReservedNames is a list of reserved enum names.
	ReservedNames() Names
	// ReservedRanges is a list of reserved ranges of enum numbers.
	ReservedRanges() EnumRanges

	isEnumDescriptor
}
type isEnumDescriptor interface{ ProtoType(EnumDescriptor) }

// EnumType encapsulates an [EnumDescriptor] with a concrete Go implementation.
type EnumType interface {
	// New returns an instance of this enum type with its value set to n.
	New(n EnumNumber) Enum

	// Descriptor returns the enum descriptor.
	//
	// Invariant: t.Descriptor() == t.New(0).Descriptor()
	Descriptor() EnumDescriptor
}

// EnumDescriptors is a list of enum declarations.
type EnumDescriptors interface {
	// Len reports the number of enum types.
	Len() int
	// Get returns the ith EnumDescriptor. It panics if out of bounds.
	Get(i int) EnumDescriptor
	// ByName returns the EnumDescriptor for an enum named s.
	// It returns nil if not found.
	ByName(s Name) EnumDescriptor

	doNotImplement
}

// EnumValueDescriptor describes an enum value and
// corresponds with the google.protobuf.EnumValueDescriptorProto message.
//
// All other proto declarations are in the namespace of the parent.
// However, enum values do not follow this rule and are within the namespace
// of the parent's parent (i.e., they are a sibling of the containing enum).
// Thus, a value named "FOO_VALUE" declared within an enum uniquely identified
// as "proto.package.MyEnum" has a full name of "proto.package.FOO_VALUE".
type EnumValueDescriptor interface {
	Descriptor

	// Number returns the enum value as an integer.
	Number() EnumNumber

	isEnumValueDescriptor
}
type isEnumValueDescriptor interface{ ProtoType(EnumValueDescriptor) }

// EnumValueDescriptors is a list of enum value declarations.
type EnumValueDescriptors interface {
	// Len reports the number of enum values.
	Len() int
	// Get returns the ith EnumValueDescriptor. It panics if out of bounds.
	Get(i int) EnumValueDescriptor
	// ByName returns the EnumValueDescriptor for the enum value named s.
	// It returns nil if not found.
	ByName(s Name) EnumValueDescriptor
	// ByNumber returns the EnumValueDescriptor for the enum value numbered n.
	// If multiple have the same number, the first one defined is returned
	// It returns nil if not found.
	ByNumber(n EnumNumber) EnumValueDescriptor

	doNotImplement
}

// ServiceDescriptor describes a service and
// corresponds with the google.protobuf.ServiceDescriptorProto message.
//
// Nested declarations: [MethodDescriptor].
type ServiceDescriptor interface {
	Descriptor

	// Methods is a list of nested message declarations.
	Methods() MethodDescriptors

	isServiceDescriptor
}
type isServiceDescriptor interface{ ProtoType(ServiceDescriptor) }

// ServiceDescriptors is a list of service declarations.
type ServiceDescriptors interface {
	// Len reports the number of services.
	Len() int
	// Get returns the ith ServiceDescriptor. It panics if out of bounds.
	Get(i int) ServiceDescriptor
	// ByName returns the ServiceDescriptor for a service named s.
	// It returns nil if not found.
	ByName(s Name) ServiceDescriptor

	doNotImplement
}

// MethodDescriptor describes a method and
// corresponds with the google.protobuf.MethodDescriptorProto message.
type MethodDescriptor interface {
	Descriptor

	// Input is the input message descriptor.
	Input() MessageDescriptor
	// Output is the output message descriptor.
	Output() MessageDescriptor
	// IsStreamingClient reports whether the client streams multiple messages.
	IsStreamingClient() bool
	// IsStreamingServer reports whether the server streams multiple messages.
	IsStreamingServer() bool

	isMethodDescriptor
}
type isMethodDescriptor interface{ ProtoType(MethodDescriptor) }

// MethodDescriptors is a list of method declarations.
type MethodDescriptors interface {
	// Len reports the number of methods.
	Len() int
	// Get returns the ith MethodDescriptor. It panics if out of bounds.
	Get(i int) MethodDescriptor
	// ByName returns the MethodDescriptor for a service method named s.
	// It returns nil if not found.
	ByName(s Name) MethodDescriptor

	doNotImplement
}
