// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocmp

import (
	"reflect"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

func reflectValueOf(v interface{}) protoreflect.Value {
	switch v := v.(type) {
	case Enum:
		return protoreflect.ValueOfEnum(v.Number())
	case Message:
		return protoreflect.ValueOfMessage(v.ProtoReflect())
	case []byte:
		return protoreflect.ValueOfBytes(v) // avoid overlap with reflect.Slice check below
	default:
		switch rv := reflect.ValueOf(v); {
		case rv.Kind() == reflect.Slice:
			return protoreflect.ValueOfList(reflectList{rv})
		case rv.Kind() == reflect.Map:
			return protoreflect.ValueOfMap(reflectMap{rv})
		default:
			return protoreflect.ValueOf(v)
		}
	}
}

type reflectMessage Message

func (m reflectMessage) stringKey(fd protoreflect.FieldDescriptor) string {
	if m.Descriptor() != fd.ContainingMessage() {
		panic("mismatching containing message")
	}
	return fd.TextName()
}

func (m reflectMessage) Descriptor() protoreflect.MessageDescriptor {
	return (Message)(m).Descriptor()
}
func (m reflectMessage) Type() protoreflect.MessageType {
	return reflectMessageType{m.Descriptor()}
}
func (m reflectMessage) New() protoreflect.Message {
	return m.Type().New()
}
func (m reflectMessage) Interface() protoreflect.ProtoMessage {
	return Message(m)
}
func (m reflectMessage) Range(f func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool) {
	// Range over populated known fields.
	fds := m.Descriptor().Fields()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		if m.Has(fd) && !f(fd, m.Get(fd)) {
			return
		}
	}

	// Range over populated extension fields.
	for _, xd := range m[messageTypeKey].(messageMeta).xds {
		if m.Has(xd) && !f(xd, m.Get(xd)) {
			return
		}
	}
}
func (m reflectMessage) Has(fd protoreflect.FieldDescriptor) bool {
	_, ok := m[m.stringKey(fd)]
	return ok
}
func (m reflectMessage) Clear(protoreflect.FieldDescriptor) {
	panic("invalid mutation of read-only message")
}
func (m reflectMessage) Get(fd protoreflect.FieldDescriptor) protoreflect.Value {
	v, ok := m[m.stringKey(fd)]
	if !ok {
		switch {
		case fd.IsList():
			return protoreflect.ValueOfList(reflectList{})
		case fd.IsMap():
			return protoreflect.ValueOfMap(reflectMap{})
		case fd.Message() != nil:
			return protoreflect.ValueOfMessage(reflectMessage{
				messageTypeKey: messageMeta{md: fd.Message()},
			})
		default:
			return fd.Default()
		}
	}

	// The transformation may leave Any messages in structured form.
	// If so, convert them back to a raw-encoded form.
	if fd.FullName() == genid.Any_Value_field_fullname {
		if m, ok := v.(Message); ok {
			b, err := proto.MarshalOptions{Deterministic: true}.Marshal(m)
			if err != nil {
				panic("BUG: " + err.Error())
			}
			return protoreflect.ValueOfBytes(b)
		}
	}

	return reflectValueOf(v)
}
func (m reflectMessage) Set(protoreflect.FieldDescriptor, protoreflect.Value) {
	panic("invalid mutation of read-only message")
}
func (m reflectMessage) Mutable(fd protoreflect.FieldDescriptor) protoreflect.Value {
	panic("invalid mutation of read-only message")
}
func (m reflectMessage) NewField(protoreflect.FieldDescriptor) protoreflect.Value {
	panic("not implemented")
}
func (m reflectMessage) WhichOneof(od protoreflect.OneofDescriptor) protoreflect.FieldDescriptor {
	if m.Descriptor().Oneofs().ByName(od.Name()) != od {
		panic("oneof descriptor does not belong to this message")
	}
	fds := od.Fields()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		if _, ok := m[m.stringKey(fd)]; ok {
			return fd
		}
	}
	return nil
}
func (m reflectMessage) GetUnknown() protoreflect.RawFields {
	var nums []protoreflect.FieldNumber
	for k := range m {
		if len(strings.Trim(k, "0123456789")) == 0 {
			n, _ := strconv.ParseUint(k, 10, 32)
			nums = append(nums, protoreflect.FieldNumber(n))
		}
	}
	sort.Slice(nums, func(i, j int) bool { return nums[i] < nums[j] })

	var raw protoreflect.RawFields
	for _, num := range nums {
		b, _ := m[strconv.FormatUint(uint64(num), 10)].(protoreflect.RawFields)
		raw = append(raw, b...)
	}
	return raw
}
func (m reflectMessage) SetUnknown(protoreflect.RawFields) {
	panic("invalid mutation of read-only message")
}
func (m reflectMessage) IsValid() bool {
	invalid, _ := m[messageInvalidKey].(bool)
	return !invalid
}
func (m reflectMessage) ProtoMethods() *protoiface.Methods {
	return nil
}

type reflectMessageType struct{ protoreflect.MessageDescriptor }

func (t reflectMessageType) New() protoreflect.Message {
	panic("not implemented")
}
func (t reflectMessageType) Zero() protoreflect.Message {
	panic("not implemented")
}
func (t reflectMessageType) Descriptor() protoreflect.MessageDescriptor {
	return t.MessageDescriptor
}

type reflectList struct{ v reflect.Value }

func (ls reflectList) Len() int {
	if !ls.IsValid() {
		return 0
	}
	return ls.v.Len()
}
func (ls reflectList) Get(i int) protoreflect.Value {
	return reflectValueOf(ls.v.Index(i).Interface())
}
func (ls reflectList) Set(int, protoreflect.Value) {
	panic("invalid mutation of read-only list")
}
func (ls reflectList) Append(protoreflect.Value) {
	panic("invalid mutation of read-only list")
}
func (ls reflectList) AppendMutable() protoreflect.Value {
	panic("invalid mutation of read-only list")
}
func (ls reflectList) Truncate(int) {
	panic("invalid mutation of read-only list")
}
func (ls reflectList) NewElement() protoreflect.Value {
	panic("not implemented")
}
func (ls reflectList) IsValid() bool {
	return ls.v.IsValid()
}

type reflectMap struct{ v reflect.Value }

func (ms reflectMap) Len() int {
	if !ms.IsValid() {
		return 0
	}
	return ms.v.Len()
}
func (ms reflectMap) Range(f func(protoreflect.MapKey, protoreflect.Value) bool) {
	if !ms.IsValid() {
		return
	}
	ks := ms.v.MapKeys()
	for _, k := range ks {
		pk := reflectValueOf(k.Interface()).MapKey()
		pv := reflectValueOf(ms.v.MapIndex(k).Interface())
		if !f(pk, pv) {
			return
		}
	}
}
func (ms reflectMap) Has(k protoreflect.MapKey) bool {
	if !ms.IsValid() {
		return false
	}
	return ms.v.MapIndex(reflect.ValueOf(k.Interface())).IsValid()
}
func (ms reflectMap) Clear(protoreflect.MapKey) {
	panic("invalid mutation of read-only list")
}
func (ms reflectMap) Get(k protoreflect.MapKey) protoreflect.Value {
	if !ms.IsValid() {
		return protoreflect.Value{}
	}
	v := ms.v.MapIndex(reflect.ValueOf(k.Interface()))
	if !v.IsValid() {
		return protoreflect.Value{}
	}
	return reflectValueOf(v.Interface())
}
func (ms reflectMap) Set(protoreflect.MapKey, protoreflect.Value) {
	panic("invalid mutation of read-only list")
}
func (ms reflectMap) Mutable(k protoreflect.MapKey) protoreflect.Value {
	panic("invalid mutation of read-only list")
}
func (ms reflectMap) NewValue() protoreflect.Value {
	panic("not implemented")
}
func (ms reflectMap) IsValid() bool {
	return ms.v.IsValid()
}
