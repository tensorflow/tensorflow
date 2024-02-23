// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import (
	"fmt"
	"math"
)

// Value is a union where only one Go type may be set at a time.
// The Value is used to represent all possible values a field may take.
// The following shows which Go type is used to represent each proto [Kind]:
//
//	╔════════════╤═════════════════════════════════════╗
//	║ Go type    │ Protobuf kind                       ║
//	╠════════════╪═════════════════════════════════════╣
//	║ bool       │ BoolKind                            ║
//	║ int32      │ Int32Kind, Sint32Kind, Sfixed32Kind ║
//	║ int64      │ Int64Kind, Sint64Kind, Sfixed64Kind ║
//	║ uint32     │ Uint32Kind, Fixed32Kind             ║
//	║ uint64     │ Uint64Kind, Fixed64Kind             ║
//	║ float32    │ FloatKind                           ║
//	║ float64    │ DoubleKind                          ║
//	║ string     │ StringKind                          ║
//	║ []byte     │ BytesKind                           ║
//	║ EnumNumber │ EnumKind                            ║
//	║ Message    │ MessageKind, GroupKind              ║
//	╚════════════╧═════════════════════════════════════╝
//
// Multiple protobuf Kinds may be represented by a single Go type if the type
// can losslessly represent the information for the proto kind. For example,
// [Int64Kind], [Sint64Kind], and [Sfixed64Kind] are all represented by int64,
// but use different integer encoding methods.
//
// The [List] or [Map] types are used if the field cardinality is repeated.
// A field is a [List] if [FieldDescriptor.IsList] reports true.
// A field is a [Map] if [FieldDescriptor.IsMap] reports true.
//
// Converting to/from a Value and a concrete Go value panics on type mismatch.
// For example, [ValueOf]("hello").Int() panics because this attempts to
// retrieve an int64 from a string.
//
// [List], [Map], and [Message] Values are called "composite" values.
//
// A composite Value may alias (reference) memory at some location,
// such that changes to the Value updates the that location.
// A composite value acquired with a Mutable method, such as [Message.Mutable],
// always references the source object.
//
// For example:
//
//	// Append a 0 to a "repeated int32" field.
//	// Since the Value returned by Mutable is guaranteed to alias
//	// the source message, modifying the Value modifies the message.
//	message.Mutable(fieldDesc).List().Append(protoreflect.ValueOfInt32(0))
//
//	// Assign [0] to a "repeated int32" field by creating a new Value,
//	// modifying it, and assigning it.
//	list := message.NewField(fieldDesc).List()
//	list.Append(protoreflect.ValueOfInt32(0))
//	message.Set(fieldDesc, list)
//	// ERROR: Since it is not defined whether Set aliases the source,
//	// appending to the List here may or may not modify the message.
//	list.Append(protoreflect.ValueOfInt32(0))
//
// Some operations, such as [Message.Get], may return an "empty, read-only"
// composite Value. Modifying an empty, read-only value panics.
type Value value

// The protoreflect API uses a custom Value union type instead of interface{}
// to keep the future open for performance optimizations. Using an interface{}
// always incurs an allocation for primitives (e.g., int64) since it needs to
// be boxed on the heap (as interfaces can only contain pointers natively).
// Instead, we represent the Value union as a flat struct that internally keeps
// track of which type is set. Using unsafe, the Value union can be reduced
// down to 24B, which is identical in size to a slice.
//
// The latest compiler (Go1.11) currently suffers from some limitations:
//	• With inlining, the compiler should be able to statically prove that
//	only one of these switch cases are taken and inline one specific case.
//	See https://golang.org/issue/22310.

// ValueOf returns a Value initialized with the concrete value stored in v.
// This panics if the type does not match one of the allowed types in the
// Value union.
func ValueOf(v interface{}) Value {
	switch v := v.(type) {
	case nil:
		return Value{}
	case bool:
		return ValueOfBool(v)
	case int32:
		return ValueOfInt32(v)
	case int64:
		return ValueOfInt64(v)
	case uint32:
		return ValueOfUint32(v)
	case uint64:
		return ValueOfUint64(v)
	case float32:
		return ValueOfFloat32(v)
	case float64:
		return ValueOfFloat64(v)
	case string:
		return ValueOfString(v)
	case []byte:
		return ValueOfBytes(v)
	case EnumNumber:
		return ValueOfEnum(v)
	case Message, List, Map:
		return valueOfIface(v)
	case ProtoMessage:
		panic(fmt.Sprintf("invalid proto.Message(%T) type, expected a protoreflect.Message type", v))
	default:
		panic(fmt.Sprintf("invalid type: %T", v))
	}
}

// ValueOfBool returns a new boolean value.
func ValueOfBool(v bool) Value {
	if v {
		return Value{typ: boolType, num: 1}
	} else {
		return Value{typ: boolType, num: 0}
	}
}

// ValueOfInt32 returns a new int32 value.
func ValueOfInt32(v int32) Value {
	return Value{typ: int32Type, num: uint64(v)}
}

// ValueOfInt64 returns a new int64 value.
func ValueOfInt64(v int64) Value {
	return Value{typ: int64Type, num: uint64(v)}
}

// ValueOfUint32 returns a new uint32 value.
func ValueOfUint32(v uint32) Value {
	return Value{typ: uint32Type, num: uint64(v)}
}

// ValueOfUint64 returns a new uint64 value.
func ValueOfUint64(v uint64) Value {
	return Value{typ: uint64Type, num: v}
}

// ValueOfFloat32 returns a new float32 value.
func ValueOfFloat32(v float32) Value {
	return Value{typ: float32Type, num: uint64(math.Float64bits(float64(v)))}
}

// ValueOfFloat64 returns a new float64 value.
func ValueOfFloat64(v float64) Value {
	return Value{typ: float64Type, num: uint64(math.Float64bits(float64(v)))}
}

// ValueOfString returns a new string value.
func ValueOfString(v string) Value {
	return valueOfString(v)
}

// ValueOfBytes returns a new bytes value.
func ValueOfBytes(v []byte) Value {
	return valueOfBytes(v[:len(v):len(v)])
}

// ValueOfEnum returns a new enum value.
func ValueOfEnum(v EnumNumber) Value {
	return Value{typ: enumType, num: uint64(v)}
}

// ValueOfMessage returns a new Message value.
func ValueOfMessage(v Message) Value {
	return valueOfIface(v)
}

// ValueOfList returns a new List value.
func ValueOfList(v List) Value {
	return valueOfIface(v)
}

// ValueOfMap returns a new Map value.
func ValueOfMap(v Map) Value {
	return valueOfIface(v)
}

// IsValid reports whether v is populated with a value.
func (v Value) IsValid() bool {
	return v.typ != nilType
}

// Interface returns v as an interface{}.
//
// Invariant: v == ValueOf(v).Interface()
func (v Value) Interface() interface{} {
	switch v.typ {
	case nilType:
		return nil
	case boolType:
		return v.Bool()
	case int32Type:
		return int32(v.Int())
	case int64Type:
		return int64(v.Int())
	case uint32Type:
		return uint32(v.Uint())
	case uint64Type:
		return uint64(v.Uint())
	case float32Type:
		return float32(v.Float())
	case float64Type:
		return float64(v.Float())
	case stringType:
		return v.String()
	case bytesType:
		return v.Bytes()
	case enumType:
		return v.Enum()
	default:
		return v.getIface()
	}
}

func (v Value) typeName() string {
	switch v.typ {
	case nilType:
		return "nil"
	case boolType:
		return "bool"
	case int32Type:
		return "int32"
	case int64Type:
		return "int64"
	case uint32Type:
		return "uint32"
	case uint64Type:
		return "uint64"
	case float32Type:
		return "float32"
	case float64Type:
		return "float64"
	case stringType:
		return "string"
	case bytesType:
		return "bytes"
	case enumType:
		return "enum"
	default:
		switch v := v.getIface().(type) {
		case Message:
			return "message"
		case List:
			return "list"
		case Map:
			return "map"
		default:
			return fmt.Sprintf("<unknown: %T>", v)
		}
	}
}

func (v Value) panicMessage(what string) string {
	return fmt.Sprintf("type mismatch: cannot convert %v to %s", v.typeName(), what)
}

// Bool returns v as a bool and panics if the type is not a bool.
func (v Value) Bool() bool {
	switch v.typ {
	case boolType:
		return v.num > 0
	default:
		panic(v.panicMessage("bool"))
	}
}

// Int returns v as a int64 and panics if the type is not a int32 or int64.
func (v Value) Int() int64 {
	switch v.typ {
	case int32Type, int64Type:
		return int64(v.num)
	default:
		panic(v.panicMessage("int"))
	}
}

// Uint returns v as a uint64 and panics if the type is not a uint32 or uint64.
func (v Value) Uint() uint64 {
	switch v.typ {
	case uint32Type, uint64Type:
		return uint64(v.num)
	default:
		panic(v.panicMessage("uint"))
	}
}

// Float returns v as a float64 and panics if the type is not a float32 or float64.
func (v Value) Float() float64 {
	switch v.typ {
	case float32Type, float64Type:
		return math.Float64frombits(uint64(v.num))
	default:
		panic(v.panicMessage("float"))
	}
}

// String returns v as a string. Since this method implements [fmt.Stringer],
// this returns the formatted string value for any non-string type.
func (v Value) String() string {
	switch v.typ {
	case stringType:
		return v.getString()
	default:
		return fmt.Sprint(v.Interface())
	}
}

// Bytes returns v as a []byte and panics if the type is not a []byte.
func (v Value) Bytes() []byte {
	switch v.typ {
	case bytesType:
		return v.getBytes()
	default:
		panic(v.panicMessage("bytes"))
	}
}

// Enum returns v as a [EnumNumber] and panics if the type is not a [EnumNumber].
func (v Value) Enum() EnumNumber {
	switch v.typ {
	case enumType:
		return EnumNumber(v.num)
	default:
		panic(v.panicMessage("enum"))
	}
}

// Message returns v as a [Message] and panics if the type is not a [Message].
func (v Value) Message() Message {
	switch vi := v.getIface().(type) {
	case Message:
		return vi
	default:
		panic(v.panicMessage("message"))
	}
}

// List returns v as a [List] and panics if the type is not a [List].
func (v Value) List() List {
	switch vi := v.getIface().(type) {
	case List:
		return vi
	default:
		panic(v.panicMessage("list"))
	}
}

// Map returns v as a [Map] and panics if the type is not a [Map].
func (v Value) Map() Map {
	switch vi := v.getIface().(type) {
	case Map:
		return vi
	default:
		panic(v.panicMessage("map"))
	}
}

// MapKey returns v as a [MapKey] and panics for invalid [MapKey] types.
func (v Value) MapKey() MapKey {
	switch v.typ {
	case boolType, int32Type, int64Type, uint32Type, uint64Type, stringType:
		return MapKey(v)
	default:
		panic(v.panicMessage("map key"))
	}
}

// MapKey is used to index maps, where the Go type of the MapKey must match
// the specified key [Kind] (see [MessageDescriptor.IsMapEntry]).
// The following shows what Go type is used to represent each proto [Kind]:
//
//	╔═════════╤═════════════════════════════════════╗
//	║ Go type │ Protobuf kind                       ║
//	╠═════════╪═════════════════════════════════════╣
//	║ bool    │ BoolKind                            ║
//	║ int32   │ Int32Kind, Sint32Kind, Sfixed32Kind ║
//	║ int64   │ Int64Kind, Sint64Kind, Sfixed64Kind ║
//	║ uint32  │ Uint32Kind, Fixed32Kind             ║
//	║ uint64  │ Uint64Kind, Fixed64Kind             ║
//	║ string  │ StringKind                          ║
//	╚═════════╧═════════════════════════════════════╝
//
// A MapKey is constructed and accessed through a [Value]:
//
//	k := ValueOf("hash").MapKey() // convert string to MapKey
//	s := k.String()               // convert MapKey to string
//
// The MapKey is a strict subset of valid types used in [Value];
// converting a [Value] to a MapKey with an invalid type panics.
type MapKey value

// IsValid reports whether k is populated with a value.
func (k MapKey) IsValid() bool {
	return Value(k).IsValid()
}

// Interface returns k as an interface{}.
func (k MapKey) Interface() interface{} {
	return Value(k).Interface()
}

// Bool returns k as a bool and panics if the type is not a bool.
func (k MapKey) Bool() bool {
	return Value(k).Bool()
}

// Int returns k as a int64 and panics if the type is not a int32 or int64.
func (k MapKey) Int() int64 {
	return Value(k).Int()
}

// Uint returns k as a uint64 and panics if the type is not a uint32 or uint64.
func (k MapKey) Uint() uint64 {
	return Value(k).Uint()
}

// String returns k as a string. Since this method implements [fmt.Stringer],
// this returns the formatted string value for any non-string type.
func (k MapKey) String() string {
	return Value(k).String()
}

// Value returns k as a [Value].
func (k MapKey) Value() Value {
	return Value(k)
}
