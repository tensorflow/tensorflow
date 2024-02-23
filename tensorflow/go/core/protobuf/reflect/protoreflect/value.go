// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import "google.golang.org/protobuf/encoding/protowire"

// Enum is a reflection interface for a concrete enum value,
// which provides type information and a getter for the enum number.
// Enum does not provide a mutable API since enums are commonly backed by
// Go constants, which are not addressable.
type Enum interface {
	// Descriptor returns enum descriptor, which contains only the protobuf
	// type information for the enum.
	Descriptor() EnumDescriptor

	// Type returns the enum type, which encapsulates both Go and protobuf
	// type information. If the Go type information is not needed,
	// it is recommended that the enum descriptor be used instead.
	Type() EnumType

	// Number returns the enum value as an integer.
	Number() EnumNumber
}

// Message is a reflective interface for a concrete message value,
// encapsulating both type and value information for the message.
//
// Accessor/mutators for individual fields are keyed by [FieldDescriptor].
// For non-extension fields, the descriptor must exactly match the
// field known by the parent message.
// For extension fields, the descriptor must implement [ExtensionTypeDescriptor],
// extend the parent message (i.e., have the same message [FullName]), and
// be within the parent's extension range.
//
// Each field [Value] can be a scalar or a composite type ([Message], [List], or [Map]).
// See [Value] for the Go types associated with a [FieldDescriptor].
// Providing a [Value] that is invalid or of an incorrect type panics.
type Message interface {
	// Descriptor returns message descriptor, which contains only the protobuf
	// type information for the message.
	Descriptor() MessageDescriptor

	// Type returns the message type, which encapsulates both Go and protobuf
	// type information. If the Go type information is not needed,
	// it is recommended that the message descriptor be used instead.
	Type() MessageType

	// New returns a newly allocated and mutable empty message.
	New() Message

	// Interface unwraps the message reflection interface and
	// returns the underlying ProtoMessage interface.
	Interface() ProtoMessage

	// Range iterates over every populated field in an undefined order,
	// calling f for each field descriptor and value encountered.
	// Range returns immediately if f returns false.
	// While iterating, mutating operations may only be performed
	// on the current field descriptor.
	Range(f func(FieldDescriptor, Value) bool)

	// Has reports whether a field is populated.
	//
	// Some fields have the property of nullability where it is possible to
	// distinguish between the default value of a field and whether the field
	// was explicitly populated with the default value. Singular message fields,
	// member fields of a oneof, and proto2 scalar fields are nullable. Such
	// fields are populated only if explicitly set.
	//
	// In other cases (aside from the nullable cases above),
	// a proto3 scalar field is populated if it contains a non-zero value, and
	// a repeated field is populated if it is non-empty.
	Has(FieldDescriptor) bool

	// Clear clears the field such that a subsequent Has call reports false.
	//
	// Clearing an extension field clears both the extension type and value
	// associated with the given field number.
	//
	// Clear is a mutating operation and unsafe for concurrent use.
	Clear(FieldDescriptor)

	// Get retrieves the value for a field.
	//
	// For unpopulated scalars, it returns the default value, where
	// the default value of a bytes scalar is guaranteed to be a copy.
	// For unpopulated composite types, it returns an empty, read-only view
	// of the value; to obtain a mutable reference, use Mutable.
	Get(FieldDescriptor) Value

	// Set stores the value for a field.
	//
	// For a field belonging to a oneof, it implicitly clears any other field
	// that may be currently set within the same oneof.
	// For extension fields, it implicitly stores the provided ExtensionType.
	// When setting a composite type, it is unspecified whether the stored value
	// aliases the source's memory in any way. If the composite value is an
	// empty, read-only value, then it panics.
	//
	// Set is a mutating operation and unsafe for concurrent use.
	Set(FieldDescriptor, Value)

	// Mutable returns a mutable reference to a composite type.
	//
	// If the field is unpopulated, it may allocate a composite value.
	// For a field belonging to a oneof, it implicitly clears any other field
	// that may be currently set within the same oneof.
	// For extension fields, it implicitly stores the provided ExtensionType
	// if not already stored.
	// It panics if the field does not contain a composite type.
	//
	// Mutable is a mutating operation and unsafe for concurrent use.
	Mutable(FieldDescriptor) Value

	// NewField returns a new value that is assignable to the field
	// for the given descriptor. For scalars, this returns the default value.
	// For lists, maps, and messages, this returns a new, empty, mutable value.
	NewField(FieldDescriptor) Value

	// WhichOneof reports which field within the oneof is populated,
	// returning nil if none are populated.
	// It panics if the oneof descriptor does not belong to this message.
	WhichOneof(OneofDescriptor) FieldDescriptor

	// GetUnknown retrieves the entire list of unknown fields.
	// The caller may only mutate the contents of the RawFields
	// if the mutated bytes are stored back into the message with SetUnknown.
	GetUnknown() RawFields

	// SetUnknown stores an entire list of unknown fields.
	// The raw fields must be syntactically valid according to the wire format.
	// An implementation may panic if this is not the case.
	// Once stored, the caller must not mutate the content of the RawFields.
	// An empty RawFields may be passed to clear the fields.
	//
	// SetUnknown is a mutating operation and unsafe for concurrent use.
	SetUnknown(RawFields)

	// IsValid reports whether the message is valid.
	//
	// An invalid message is an empty, read-only value.
	//
	// An invalid message often corresponds to a nil pointer of the concrete
	// message type, but the details are implementation dependent.
	// Validity is not part of the protobuf data model, and may not
	// be preserved in marshaling or other operations.
	IsValid() bool

	// ProtoMethods returns optional fast-path implementations of various operations.
	// This method may return nil.
	//
	// The returned methods type is identical to
	// google.golang.org/protobuf/runtime/protoiface.Methods.
	// Consult the protoiface package documentation for details.
	ProtoMethods() *methods
}

// RawFields is the raw bytes for an ordered sequence of fields.
// Each field contains both the tag (representing field number and wire type),
// and also the wire data itself.
type RawFields []byte

// IsValid reports whether b is syntactically correct wire format.
func (b RawFields) IsValid() bool {
	for len(b) > 0 {
		_, _, n := protowire.ConsumeField(b)
		if n < 0 {
			return false
		}
		b = b[n:]
	}
	return true
}

// List is a zero-indexed, ordered list.
// The element [Value] type is determined by [FieldDescriptor.Kind].
// Providing a [Value] that is invalid or of an incorrect type panics.
type List interface {
	// Len reports the number of entries in the List.
	// Get, Set, and Truncate panic with out of bound indexes.
	Len() int

	// Get retrieves the value at the given index.
	// It never returns an invalid value.
	Get(int) Value

	// Set stores a value for the given index.
	// When setting a composite type, it is unspecified whether the set
	// value aliases the source's memory in any way.
	//
	// Set is a mutating operation and unsafe for concurrent use.
	Set(int, Value)

	// Append appends the provided value to the end of the list.
	// When appending a composite type, it is unspecified whether the appended
	// value aliases the source's memory in any way.
	//
	// Append is a mutating operation and unsafe for concurrent use.
	Append(Value)

	// AppendMutable appends a new, empty, mutable message value to the end
	// of the list and returns it.
	// It panics if the list does not contain a message type.
	AppendMutable() Value

	// Truncate truncates the list to a smaller length.
	//
	// Truncate is a mutating operation and unsafe for concurrent use.
	Truncate(int)

	// NewElement returns a new value for a list element.
	// For enums, this returns the first enum value.
	// For other scalars, this returns the zero value.
	// For messages, this returns a new, empty, mutable value.
	NewElement() Value

	// IsValid reports whether the list is valid.
	//
	// An invalid list is an empty, read-only value.
	//
	// Validity is not part of the protobuf data model, and may not
	// be preserved in marshaling or other operations.
	IsValid() bool
}

// Map is an unordered, associative map.
// The entry [MapKey] type is determined by [FieldDescriptor.MapKey].Kind.
// The entry [Value] type is determined by [FieldDescriptor.MapValue].Kind.
// Providing a [MapKey] or [Value] that is invalid or of an incorrect type panics.
type Map interface {
	// Len reports the number of elements in the map.
	Len() int

	// Range iterates over every map entry in an undefined order,
	// calling f for each key and value encountered.
	// Range calls f Len times unless f returns false, which stops iteration.
	// While iterating, mutating operations may only be performed
	// on the current map key.
	Range(f func(MapKey, Value) bool)

	// Has reports whether an entry with the given key is in the map.
	Has(MapKey) bool

	// Clear clears the entry associated with they given key.
	// The operation does nothing if there is no entry associated with the key.
	//
	// Clear is a mutating operation and unsafe for concurrent use.
	Clear(MapKey)

	// Get retrieves the value for an entry with the given key.
	// It returns an invalid value for non-existent entries.
	Get(MapKey) Value

	// Set stores the value for an entry with the given key.
	// It panics when given a key or value that is invalid or the wrong type.
	// When setting a composite type, it is unspecified whether the set
	// value aliases the source's memory in any way.
	//
	// Set is a mutating operation and unsafe for concurrent use.
	Set(MapKey, Value)

	// Mutable retrieves a mutable reference to the entry for the given key.
	// If no entry exists for the key, it creates a new, empty, mutable value
	// and stores it as the entry for the key.
	// It panics if the map value is not a message.
	Mutable(MapKey) Value

	// NewValue returns a new value assignable as a map value.
	// For enums, this returns the first enum value.
	// For other scalars, this returns the zero value.
	// For messages, this returns a new, empty, mutable value.
	NewValue() Value

	// IsValid reports whether the map is valid.
	//
	// An invalid map is an empty, read-only value.
	//
	// An invalid message often corresponds to a nil Go map value,
	// but the details are implementation dependent.
	// Validity is not part of the protobuf data model, and may not
	// be preserved in marshaling or other operations.
	IsValid() bool
}
