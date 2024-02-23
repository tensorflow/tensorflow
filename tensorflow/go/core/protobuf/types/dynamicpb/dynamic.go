// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dynamicpb creates protocol buffer messages using runtime type information.
package dynamicpb

import (
	"math"

	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
	"google.golang.org/protobuf/runtime/protoimpl"
)

// enum is a dynamic protoreflect.Enum.
type enum struct {
	num protoreflect.EnumNumber
	typ protoreflect.EnumType
}

func (e enum) Descriptor() protoreflect.EnumDescriptor { return e.typ.Descriptor() }
func (e enum) Type() protoreflect.EnumType             { return e.typ }
func (e enum) Number() protoreflect.EnumNumber         { return e.num }

// enumType is a dynamic protoreflect.EnumType.
type enumType struct {
	desc protoreflect.EnumDescriptor
}

// NewEnumType creates a new EnumType with the provided descriptor.
//
// EnumTypes created by this package are equal if their descriptors are equal.
// That is, if ed1 == ed2, then NewEnumType(ed1) == NewEnumType(ed2).
//
// Enum values created by the EnumType are equal if their numbers are equal.
func NewEnumType(desc protoreflect.EnumDescriptor) protoreflect.EnumType {
	return enumType{desc}
}

func (et enumType) New(n protoreflect.EnumNumber) protoreflect.Enum { return enum{n, et} }
func (et enumType) Descriptor() protoreflect.EnumDescriptor         { return et.desc }

// extensionType is a dynamic protoreflect.ExtensionType.
type extensionType struct {
	desc extensionTypeDescriptor
}

// A Message is a dynamically constructed protocol buffer message.
//
// Message implements the [google.golang.org/protobuf/proto.Message] interface,
// and may be used with all  standard proto package functions
// such as Marshal, Unmarshal, and so forth.
//
// Message also implements the [protoreflect.Message] interface.
// See the [protoreflect] package documentation for that interface for how to
// get and set fields and otherwise interact with the contents of a Message.
//
// Reflection API functions which construct messages, such as NewField,
// return new dynamic messages of the appropriate type. Functions which take
// messages, such as Set for a message-value field, will accept any message
// with a compatible type.
//
// Operations which modify a Message are not safe for concurrent use.
type Message struct {
	typ     messageType
	known   map[protoreflect.FieldNumber]protoreflect.Value
	ext     map[protoreflect.FieldNumber]protoreflect.FieldDescriptor
	unknown protoreflect.RawFields
}

var (
	_ protoreflect.Message      = (*Message)(nil)
	_ protoreflect.ProtoMessage = (*Message)(nil)
	_ protoiface.MessageV1      = (*Message)(nil)
)

// NewMessage creates a new message with the provided descriptor.
func NewMessage(desc protoreflect.MessageDescriptor) *Message {
	return &Message{
		typ:   messageType{desc},
		known: make(map[protoreflect.FieldNumber]protoreflect.Value),
		ext:   make(map[protoreflect.FieldNumber]protoreflect.FieldDescriptor),
	}
}

// ProtoMessage implements the legacy message interface.
func (m *Message) ProtoMessage() {}

// ProtoReflect implements the [protoreflect.ProtoMessage] interface.
func (m *Message) ProtoReflect() protoreflect.Message {
	return m
}

// String returns a string representation of a message.
func (m *Message) String() string {
	return protoimpl.X.MessageStringOf(m)
}

// Reset clears the message to be empty, but preserves the dynamic message type.
func (m *Message) Reset() {
	m.known = make(map[protoreflect.FieldNumber]protoreflect.Value)
	m.ext = make(map[protoreflect.FieldNumber]protoreflect.FieldDescriptor)
	m.unknown = nil
}

// Descriptor returns the message descriptor.
func (m *Message) Descriptor() protoreflect.MessageDescriptor {
	return m.typ.desc
}

// Type returns the message type.
func (m *Message) Type() protoreflect.MessageType {
	return m.typ
}

// New returns a newly allocated empty message with the same descriptor.
// See [protoreflect.Message] for details.
func (m *Message) New() protoreflect.Message {
	return m.Type().New()
}

// Interface returns the message.
// See [protoreflect.Message] for details.
func (m *Message) Interface() protoreflect.ProtoMessage {
	return m
}

// ProtoMethods is an internal detail of the [protoreflect.Message] interface.
// Users should never call this directly.
func (m *Message) ProtoMethods() *protoiface.Methods {
	return nil
}

// Range visits every populated field in undefined order.
// See [protoreflect.Message] for details.
func (m *Message) Range(f func(protoreflect.FieldDescriptor, protoreflect.Value) bool) {
	for num, v := range m.known {
		fd := m.ext[num]
		if fd == nil {
			fd = m.Descriptor().Fields().ByNumber(num)
		}
		if !isSet(fd, v) {
			continue
		}
		if !f(fd, v) {
			return
		}
	}
}

// Has reports whether a field is populated.
// See [protoreflect.Message] for details.
func (m *Message) Has(fd protoreflect.FieldDescriptor) bool {
	m.checkField(fd)
	if fd.IsExtension() && m.ext[fd.Number()] != fd {
		return false
	}
	v, ok := m.known[fd.Number()]
	if !ok {
		return false
	}
	return isSet(fd, v)
}

// Clear clears a field.
// See [protoreflect.Message] for details.
func (m *Message) Clear(fd protoreflect.FieldDescriptor) {
	m.checkField(fd)
	num := fd.Number()
	delete(m.known, num)
	delete(m.ext, num)
}

// Get returns the value of a field.
// See [protoreflect.Message] for details.
func (m *Message) Get(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.checkField(fd)
	num := fd.Number()
	if fd.IsExtension() {
		if fd != m.ext[num] {
			return fd.(protoreflect.ExtensionTypeDescriptor).Type().Zero()
		}
		return m.known[num]
	}
	if v, ok := m.known[num]; ok {
		switch {
		case fd.IsMap():
			if v.Map().Len() > 0 {
				return v
			}
		case fd.IsList():
			if v.List().Len() > 0 {
				return v
			}
		default:
			return v
		}
	}
	switch {
	case fd.IsMap():
		return protoreflect.ValueOfMap(&dynamicMap{desc: fd})
	case fd.IsList():
		return protoreflect.ValueOfList(emptyList{desc: fd})
	case fd.Message() != nil:
		return protoreflect.ValueOfMessage(&Message{typ: messageType{fd.Message()}})
	case fd.Kind() == protoreflect.BytesKind:
		return protoreflect.ValueOfBytes(append([]byte(nil), fd.Default().Bytes()...))
	default:
		return fd.Default()
	}
}

// Mutable returns a mutable reference to a repeated, map, or message field.
// See [protoreflect.Message] for details.
func (m *Message) Mutable(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.checkField(fd)
	if !fd.IsMap() && !fd.IsList() && fd.Message() == nil {
		panic(errors.New("%v: getting mutable reference to non-composite type", fd.FullName()))
	}
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", fd.FullName()))
	}
	num := fd.Number()
	if fd.IsExtension() {
		if fd != m.ext[num] {
			m.ext[num] = fd
			m.known[num] = fd.(protoreflect.ExtensionTypeDescriptor).Type().New()
		}
		return m.known[num]
	}
	if v, ok := m.known[num]; ok {
		return v
	}
	m.clearOtherOneofFields(fd)
	m.known[num] = m.NewField(fd)
	if fd.IsExtension() {
		m.ext[num] = fd
	}
	return m.known[num]
}

// Set stores a value in a field.
// See [protoreflect.Message] for details.
func (m *Message) Set(fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	m.checkField(fd)
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", fd.FullName()))
	}
	if fd.IsExtension() {
		isValid := true
		switch {
		case !fd.(protoreflect.ExtensionTypeDescriptor).Type().IsValidValue(v):
			isValid = false
		case fd.IsList():
			isValid = v.List().IsValid()
		case fd.IsMap():
			isValid = v.Map().IsValid()
		case fd.Message() != nil:
			isValid = v.Message().IsValid()
		}
		if !isValid {
			panic(errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface()))
		}
		m.ext[fd.Number()] = fd
	} else {
		typecheck(fd, v)
	}
	m.clearOtherOneofFields(fd)
	m.known[fd.Number()] = v
}

func (m *Message) clearOtherOneofFields(fd protoreflect.FieldDescriptor) {
	od := fd.ContainingOneof()
	if od == nil {
		return
	}
	num := fd.Number()
	for i := 0; i < od.Fields().Len(); i++ {
		if n := od.Fields().Get(i).Number(); n != num {
			delete(m.known, n)
		}
	}
}

// NewField returns a new value for assignable to the field of a given descriptor.
// See [protoreflect.Message] for details.
func (m *Message) NewField(fd protoreflect.FieldDescriptor) protoreflect.Value {
	m.checkField(fd)
	switch {
	case fd.IsExtension():
		return fd.(protoreflect.ExtensionTypeDescriptor).Type().New()
	case fd.IsMap():
		return protoreflect.ValueOfMap(&dynamicMap{
			desc: fd,
			mapv: make(map[interface{}]protoreflect.Value),
		})
	case fd.IsList():
		return protoreflect.ValueOfList(&dynamicList{desc: fd})
	case fd.Message() != nil:
		return protoreflect.ValueOfMessage(NewMessage(fd.Message()).ProtoReflect())
	default:
		return fd.Default()
	}
}

// WhichOneof reports which field in a oneof is populated, returning nil if none are populated.
// See [protoreflect.Message] for details.
func (m *Message) WhichOneof(od protoreflect.OneofDescriptor) protoreflect.FieldDescriptor {
	for i := 0; i < od.Fields().Len(); i++ {
		fd := od.Fields().Get(i)
		if m.Has(fd) {
			return fd
		}
	}
	return nil
}

// GetUnknown returns the raw unknown fields.
// See [protoreflect.Message] for details.
func (m *Message) GetUnknown() protoreflect.RawFields {
	return m.unknown
}

// SetUnknown sets the raw unknown fields.
// See [protoreflect.Message] for details.
func (m *Message) SetUnknown(r protoreflect.RawFields) {
	if m.known == nil {
		panic(errors.New("%v: modification of read-only message", m.typ.desc.FullName()))
	}
	m.unknown = r
}

// IsValid reports whether the message is valid.
// See [protoreflect.Message] for details.
func (m *Message) IsValid() bool {
	return m.known != nil
}

func (m *Message) checkField(fd protoreflect.FieldDescriptor) {
	if fd.IsExtension() && fd.ContainingMessage().FullName() == m.Descriptor().FullName() {
		if _, ok := fd.(protoreflect.ExtensionTypeDescriptor); !ok {
			panic(errors.New("%v: extension field descriptor does not implement ExtensionTypeDescriptor", fd.FullName()))
		}
		return
	}
	if fd.Parent() == m.Descriptor() {
		return
	}
	fields := m.Descriptor().Fields()
	index := fd.Index()
	if index >= fields.Len() || fields.Get(index) != fd {
		panic(errors.New("%v: field descriptor does not belong to this message", fd.FullName()))
	}
}

type messageType struct {
	desc protoreflect.MessageDescriptor
}

// NewMessageType creates a new MessageType with the provided descriptor.
//
// MessageTypes created by this package are equal if their descriptors are equal.
// That is, if md1 == md2, then NewMessageType(md1) == NewMessageType(md2).
func NewMessageType(desc protoreflect.MessageDescriptor) protoreflect.MessageType {
	return messageType{desc}
}

func (mt messageType) New() protoreflect.Message                  { return NewMessage(mt.desc) }
func (mt messageType) Zero() protoreflect.Message                 { return &Message{typ: messageType{mt.desc}} }
func (mt messageType) Descriptor() protoreflect.MessageDescriptor { return mt.desc }
func (mt messageType) Enum(i int) protoreflect.EnumType {
	if ed := mt.desc.Fields().Get(i).Enum(); ed != nil {
		return NewEnumType(ed)
	}
	return nil
}
func (mt messageType) Message(i int) protoreflect.MessageType {
	if md := mt.desc.Fields().Get(i).Message(); md != nil {
		return NewMessageType(md)
	}
	return nil
}

type emptyList struct {
	desc protoreflect.FieldDescriptor
}

func (x emptyList) Len() int                     { return 0 }
func (x emptyList) Get(n int) protoreflect.Value { panic(errors.New("out of range")) }
func (x emptyList) Set(n int, v protoreflect.Value) {
	panic(errors.New("modification of immutable list"))
}
func (x emptyList) Append(v protoreflect.Value) { panic(errors.New("modification of immutable list")) }
func (x emptyList) AppendMutable() protoreflect.Value {
	panic(errors.New("modification of immutable list"))
}
func (x emptyList) Truncate(n int)                 { panic(errors.New("modification of immutable list")) }
func (x emptyList) NewElement() protoreflect.Value { return newListEntry(x.desc) }
func (x emptyList) IsValid() bool                  { return false }

type dynamicList struct {
	desc protoreflect.FieldDescriptor
	list []protoreflect.Value
}

func (x *dynamicList) Len() int {
	return len(x.list)
}

func (x *dynamicList) Get(n int) protoreflect.Value {
	return x.list[n]
}

func (x *dynamicList) Set(n int, v protoreflect.Value) {
	typecheckSingular(x.desc, v)
	x.list[n] = v
}

func (x *dynamicList) Append(v protoreflect.Value) {
	typecheckSingular(x.desc, v)
	x.list = append(x.list, v)
}

func (x *dynamicList) AppendMutable() protoreflect.Value {
	if x.desc.Message() == nil {
		panic(errors.New("%v: invalid AppendMutable on list with non-message type", x.desc.FullName()))
	}
	v := x.NewElement()
	x.Append(v)
	return v
}

func (x *dynamicList) Truncate(n int) {
	// Zero truncated elements to avoid keeping data live.
	for i := n; i < len(x.list); i++ {
		x.list[i] = protoreflect.Value{}
	}
	x.list = x.list[:n]
}

func (x *dynamicList) NewElement() protoreflect.Value {
	return newListEntry(x.desc)
}

func (x *dynamicList) IsValid() bool {
	return true
}

type dynamicMap struct {
	desc protoreflect.FieldDescriptor
	mapv map[interface{}]protoreflect.Value
}

func (x *dynamicMap) Get(k protoreflect.MapKey) protoreflect.Value { return x.mapv[k.Interface()] }
func (x *dynamicMap) Set(k protoreflect.MapKey, v protoreflect.Value) {
	typecheckSingular(x.desc.MapKey(), k.Value())
	typecheckSingular(x.desc.MapValue(), v)
	x.mapv[k.Interface()] = v
}
func (x *dynamicMap) Has(k protoreflect.MapKey) bool { return x.Get(k).IsValid() }
func (x *dynamicMap) Clear(k protoreflect.MapKey)    { delete(x.mapv, k.Interface()) }
func (x *dynamicMap) Mutable(k protoreflect.MapKey) protoreflect.Value {
	if x.desc.MapValue().Message() == nil {
		panic(errors.New("%v: invalid Mutable on map with non-message value type", x.desc.FullName()))
	}
	v := x.Get(k)
	if !v.IsValid() {
		v = x.NewValue()
		x.Set(k, v)
	}
	return v
}
func (x *dynamicMap) Len() int { return len(x.mapv) }
func (x *dynamicMap) NewValue() protoreflect.Value {
	if md := x.desc.MapValue().Message(); md != nil {
		return protoreflect.ValueOfMessage(NewMessage(md).ProtoReflect())
	}
	return x.desc.MapValue().Default()
}
func (x *dynamicMap) IsValid() bool {
	return x.mapv != nil
}

func (x *dynamicMap) Range(f func(protoreflect.MapKey, protoreflect.Value) bool) {
	for k, v := range x.mapv {
		if !f(protoreflect.ValueOf(k).MapKey(), v) {
			return
		}
	}
}

func isSet(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
	switch {
	case fd.IsMap():
		return v.Map().Len() > 0
	case fd.IsList():
		return v.List().Len() > 0
	case fd.ContainingOneof() != nil:
		return true
	case !fd.HasPresence() && !fd.IsExtension():
		switch fd.Kind() {
		case protoreflect.BoolKind:
			return v.Bool()
		case protoreflect.EnumKind:
			return v.Enum() != 0
		case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed32Kind, protoreflect.Sfixed64Kind:
			return v.Int() != 0
		case protoreflect.Uint32Kind, protoreflect.Uint64Kind, protoreflect.Fixed32Kind, protoreflect.Fixed64Kind:
			return v.Uint() != 0
		case protoreflect.FloatKind, protoreflect.DoubleKind:
			return v.Float() != 0 || math.Signbit(v.Float())
		case protoreflect.StringKind:
			return v.String() != ""
		case protoreflect.BytesKind:
			return len(v.Bytes()) > 0
		}
	}
	return true
}

func typecheck(fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	if err := typeIsValid(fd, v); err != nil {
		panic(err)
	}
}

func typeIsValid(fd protoreflect.FieldDescriptor, v protoreflect.Value) error {
	switch {
	case !v.IsValid():
		return errors.New("%v: assigning invalid value", fd.FullName())
	case fd.IsMap():
		if mapv, ok := v.Interface().(*dynamicMap); !ok || mapv.desc != fd || !mapv.IsValid() {
			return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
		}
		return nil
	case fd.IsList():
		switch list := v.Interface().(type) {
		case *dynamicList:
			if list.desc == fd && list.IsValid() {
				return nil
			}
		case emptyList:
			if list.desc == fd && list.IsValid() {
				return nil
			}
		}
		return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
	default:
		return singularTypeIsValid(fd, v)
	}
}

func typecheckSingular(fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	if err := singularTypeIsValid(fd, v); err != nil {
		panic(err)
	}
}

func singularTypeIsValid(fd protoreflect.FieldDescriptor, v protoreflect.Value) error {
	vi := v.Interface()
	var ok bool
	switch fd.Kind() {
	case protoreflect.BoolKind:
		_, ok = vi.(bool)
	case protoreflect.EnumKind:
		// We could check against the valid set of enum values, but do not.
		_, ok = vi.(protoreflect.EnumNumber)
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		_, ok = vi.(int32)
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		_, ok = vi.(uint32)
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		_, ok = vi.(int64)
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		_, ok = vi.(uint64)
	case protoreflect.FloatKind:
		_, ok = vi.(float32)
	case protoreflect.DoubleKind:
		_, ok = vi.(float64)
	case protoreflect.StringKind:
		_, ok = vi.(string)
	case protoreflect.BytesKind:
		_, ok = vi.([]byte)
	case protoreflect.MessageKind, protoreflect.GroupKind:
		var m protoreflect.Message
		m, ok = vi.(protoreflect.Message)
		if ok && m.Descriptor().FullName() != fd.Message().FullName() {
			return errors.New("%v: assigning invalid message type %v", fd.FullName(), m.Descriptor().FullName())
		}
		if dm, ok := vi.(*Message); ok && dm.known == nil {
			return errors.New("%v: assigning invalid zero-value message", fd.FullName())
		}
	}
	if !ok {
		return errors.New("%v: assigning invalid type %T", fd.FullName(), v.Interface())
	}
	return nil
}

func newListEntry(fd protoreflect.FieldDescriptor) protoreflect.Value {
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return protoreflect.ValueOfBool(false)
	case protoreflect.EnumKind:
		return protoreflect.ValueOfEnum(fd.Enum().Values().Get(0).Number())
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		return protoreflect.ValueOfInt32(0)
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		return protoreflect.ValueOfUint32(0)
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		return protoreflect.ValueOfInt64(0)
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		return protoreflect.ValueOfUint64(0)
	case protoreflect.FloatKind:
		return protoreflect.ValueOfFloat32(0)
	case protoreflect.DoubleKind:
		return protoreflect.ValueOfFloat64(0)
	case protoreflect.StringKind:
		return protoreflect.ValueOfString("")
	case protoreflect.BytesKind:
		return protoreflect.ValueOfBytes(nil)
	case protoreflect.MessageKind, protoreflect.GroupKind:
		return protoreflect.ValueOfMessage(NewMessage(fd.Message()).ProtoReflect())
	}
	panic(errors.New("%v: unknown kind %v", fd.FullName(), fd.Kind()))
}

// NewExtensionType creates a new ExtensionType with the provided descriptor.
//
// Dynamic ExtensionTypes with the same descriptor compare as equal. That is,
// if xd1 == xd2, then NewExtensionType(xd1) == NewExtensionType(xd2).
//
// The InterfaceOf and ValueOf methods of the extension type are defined as:
//
//	func (xt extensionType) ValueOf(iv interface{}) protoreflect.Value {
//		return protoreflect.ValueOf(iv)
//	}
//
//	func (xt extensionType) InterfaceOf(v protoreflect.Value) interface{} {
//		return v.Interface()
//	}
//
// The Go type used by the proto.GetExtension and proto.SetExtension functions
// is determined by these methods, and is therefore equivalent to the Go type
// used to represent a protoreflect.Value. See the protoreflect.Value
// documentation for more details.
func NewExtensionType(desc protoreflect.ExtensionDescriptor) protoreflect.ExtensionType {
	if xt, ok := desc.(protoreflect.ExtensionTypeDescriptor); ok {
		desc = xt.Descriptor()
	}
	return extensionType{extensionTypeDescriptor{desc}}
}

func (xt extensionType) New() protoreflect.Value {
	switch {
	case xt.desc.IsMap():
		return protoreflect.ValueOfMap(&dynamicMap{
			desc: xt.desc,
			mapv: make(map[interface{}]protoreflect.Value),
		})
	case xt.desc.IsList():
		return protoreflect.ValueOfList(&dynamicList{desc: xt.desc})
	case xt.desc.Message() != nil:
		return protoreflect.ValueOfMessage(NewMessage(xt.desc.Message()))
	default:
		return xt.desc.Default()
	}
}

func (xt extensionType) Zero() protoreflect.Value {
	switch {
	case xt.desc.IsMap():
		return protoreflect.ValueOfMap(&dynamicMap{desc: xt.desc})
	case xt.desc.Cardinality() == protoreflect.Repeated:
		return protoreflect.ValueOfList(emptyList{desc: xt.desc})
	case xt.desc.Message() != nil:
		return protoreflect.ValueOfMessage(&Message{typ: messageType{xt.desc.Message()}})
	default:
		return xt.desc.Default()
	}
}

func (xt extensionType) TypeDescriptor() protoreflect.ExtensionTypeDescriptor {
	return xt.desc
}

func (xt extensionType) ValueOf(iv interface{}) protoreflect.Value {
	v := protoreflect.ValueOf(iv)
	typecheck(xt.desc, v)
	return v
}

func (xt extensionType) InterfaceOf(v protoreflect.Value) interface{} {
	typecheck(xt.desc, v)
	return v.Interface()
}

func (xt extensionType) IsValidInterface(iv interface{}) bool {
	return typeIsValid(xt.desc, protoreflect.ValueOf(iv)) == nil
}

func (xt extensionType) IsValidValue(v protoreflect.Value) bool {
	return typeIsValid(xt.desc, v) == nil
}

type extensionTypeDescriptor struct {
	protoreflect.ExtensionDescriptor
}

func (xt extensionTypeDescriptor) Type() protoreflect.ExtensionType {
	return extensionType{xt}
}

func (xt extensionTypeDescriptor) Descriptor() protoreflect.ExtensionDescriptor {
	return xt.ExtensionDescriptor
}
