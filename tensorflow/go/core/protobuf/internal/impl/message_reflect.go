// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"

	"google.golang.org/protobuf/internal/detrand"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/reflect/protoreflect"
)

type reflectMessageInfo struct {
	fields map[protoreflect.FieldNumber]*fieldInfo
	oneofs map[protoreflect.Name]*oneofInfo

	// fieldTypes contains the zero value of an enum or message field.
	// For lists, it contains the element type.
	// For maps, it contains the entry value type.
	fieldTypes map[protoreflect.FieldNumber]interface{}

	// denseFields is a subset of fields where:
	//	0 < fieldDesc.Number() < len(denseFields)
	// It provides faster access to the fieldInfo, but may be incomplete.
	denseFields []*fieldInfo

	// rangeInfos is a list of all fields (not belonging to a oneof) and oneofs.
	rangeInfos []interface{} // either *fieldInfo or *oneofInfo

	getUnknown   func(pointer) protoreflect.RawFields
	setUnknown   func(pointer, protoreflect.RawFields)
	extensionMap func(pointer) *extensionMap

	nilMessage atomicNilMessage
}

// makeReflectFuncs generates the set of functions to support reflection.
func (mi *MessageInfo) makeReflectFuncs(t reflect.Type, si structInfo) {
	mi.makeKnownFieldsFunc(si)
	mi.makeUnknownFieldsFunc(t, si)
	mi.makeExtensionFieldsFunc(t, si)
	mi.makeFieldTypes(si)
}

// makeKnownFieldsFunc generates functions for operations that can be performed
// on each protobuf message field. It takes in a reflect.Type representing the
// Go struct and matches message fields with struct fields.
//
// This code assumes that the struct is well-formed and panics if there are
// any discrepancies.
func (mi *MessageInfo) makeKnownFieldsFunc(si structInfo) {
	mi.fields = map[protoreflect.FieldNumber]*fieldInfo{}
	md := mi.Desc
	fds := md.Fields()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		fs := si.fieldsByNumber[fd.Number()]
		isOneof := fd.ContainingOneof() != nil && !fd.ContainingOneof().IsSynthetic()
		if isOneof {
			fs = si.oneofsByName[fd.ContainingOneof().Name()]
		}
		var fi fieldInfo
		switch {
		case fs.Type == nil:
			fi = fieldInfoForMissing(fd) // never occurs for officially generated message types
		case isOneof:
			fi = fieldInfoForOneof(fd, fs, mi.Exporter, si.oneofWrappersByNumber[fd.Number()])
		case fd.IsMap():
			fi = fieldInfoForMap(fd, fs, mi.Exporter)
		case fd.IsList():
			fi = fieldInfoForList(fd, fs, mi.Exporter)
		case fd.IsWeak():
			fi = fieldInfoForWeakMessage(fd, si.weakOffset)
		case fd.Message() != nil:
			fi = fieldInfoForMessage(fd, fs, mi.Exporter)
		default:
			fi = fieldInfoForScalar(fd, fs, mi.Exporter)
		}
		mi.fields[fd.Number()] = &fi
	}

	mi.oneofs = map[protoreflect.Name]*oneofInfo{}
	for i := 0; i < md.Oneofs().Len(); i++ {
		od := md.Oneofs().Get(i)
		mi.oneofs[od.Name()] = makeOneofInfo(od, si, mi.Exporter)
	}

	mi.denseFields = make([]*fieldInfo, fds.Len()*2)
	for i := 0; i < fds.Len(); i++ {
		if fd := fds.Get(i); int(fd.Number()) < len(mi.denseFields) {
			mi.denseFields[fd.Number()] = mi.fields[fd.Number()]
		}
	}

	for i := 0; i < fds.Len(); {
		fd := fds.Get(i)
		if od := fd.ContainingOneof(); od != nil && !od.IsSynthetic() {
			mi.rangeInfos = append(mi.rangeInfos, mi.oneofs[od.Name()])
			i += od.Fields().Len()
		} else {
			mi.rangeInfos = append(mi.rangeInfos, mi.fields[fd.Number()])
			i++
		}
	}

	// Introduce instability to iteration order, but keep it deterministic.
	if len(mi.rangeInfos) > 1 && detrand.Bool() {
		i := detrand.Intn(len(mi.rangeInfos) - 1)
		mi.rangeInfos[i], mi.rangeInfos[i+1] = mi.rangeInfos[i+1], mi.rangeInfos[i]
	}
}

func (mi *MessageInfo) makeUnknownFieldsFunc(t reflect.Type, si structInfo) {
	switch {
	case si.unknownOffset.IsValid() && si.unknownType == unknownFieldsAType:
		// Handle as []byte.
		mi.getUnknown = func(p pointer) protoreflect.RawFields {
			if p.IsNil() {
				return nil
			}
			return *p.Apply(mi.unknownOffset).Bytes()
		}
		mi.setUnknown = func(p pointer, b protoreflect.RawFields) {
			if p.IsNil() {
				panic("invalid SetUnknown on nil Message")
			}
			*p.Apply(mi.unknownOffset).Bytes() = b
		}
	case si.unknownOffset.IsValid() && si.unknownType == unknownFieldsBType:
		// Handle as *[]byte.
		mi.getUnknown = func(p pointer) protoreflect.RawFields {
			if p.IsNil() {
				return nil
			}
			bp := p.Apply(mi.unknownOffset).BytesPtr()
			if *bp == nil {
				return nil
			}
			return **bp
		}
		mi.setUnknown = func(p pointer, b protoreflect.RawFields) {
			if p.IsNil() {
				panic("invalid SetUnknown on nil Message")
			}
			bp := p.Apply(mi.unknownOffset).BytesPtr()
			if *bp == nil {
				*bp = new([]byte)
			}
			**bp = b
		}
	default:
		mi.getUnknown = func(pointer) protoreflect.RawFields {
			return nil
		}
		mi.setUnknown = func(p pointer, _ protoreflect.RawFields) {
			if p.IsNil() {
				panic("invalid SetUnknown on nil Message")
			}
		}
	}
}

func (mi *MessageInfo) makeExtensionFieldsFunc(t reflect.Type, si structInfo) {
	if si.extensionOffset.IsValid() {
		mi.extensionMap = func(p pointer) *extensionMap {
			if p.IsNil() {
				return (*extensionMap)(nil)
			}
			v := p.Apply(si.extensionOffset).AsValueOf(extensionFieldsType)
			return (*extensionMap)(v.Interface().(*map[int32]ExtensionField))
		}
	} else {
		mi.extensionMap = func(pointer) *extensionMap {
			return (*extensionMap)(nil)
		}
	}
}
func (mi *MessageInfo) makeFieldTypes(si structInfo) {
	md := mi.Desc
	fds := md.Fields()
	for i := 0; i < fds.Len(); i++ {
		var ft reflect.Type
		fd := fds.Get(i)
		fs := si.fieldsByNumber[fd.Number()]
		isOneof := fd.ContainingOneof() != nil && !fd.ContainingOneof().IsSynthetic()
		if isOneof {
			fs = si.oneofsByName[fd.ContainingOneof().Name()]
		}
		var isMessage bool
		switch {
		case fs.Type == nil:
			continue // never occurs for officially generated message types
		case isOneof:
			if fd.Enum() != nil || fd.Message() != nil {
				ft = si.oneofWrappersByNumber[fd.Number()].Field(0).Type
			}
		case fd.IsMap():
			if fd.MapValue().Enum() != nil || fd.MapValue().Message() != nil {
				ft = fs.Type.Elem()
			}
			isMessage = fd.MapValue().Message() != nil
		case fd.IsList():
			if fd.Enum() != nil || fd.Message() != nil {
				ft = fs.Type.Elem()
			}
			isMessage = fd.Message() != nil
		case fd.Enum() != nil:
			ft = fs.Type
			if fd.HasPresence() && ft.Kind() == reflect.Ptr {
				ft = ft.Elem()
			}
		case fd.Message() != nil:
			ft = fs.Type
			if fd.IsWeak() {
				ft = nil
			}
			isMessage = true
		}
		if isMessage && ft != nil && ft.Kind() != reflect.Ptr {
			ft = reflect.PtrTo(ft) // never occurs for officially generated message types
		}
		if ft != nil {
			if mi.fieldTypes == nil {
				mi.fieldTypes = make(map[protoreflect.FieldNumber]interface{})
			}
			mi.fieldTypes[fd.Number()] = reflect.Zero(ft).Interface()
		}
	}
}

type extensionMap map[int32]ExtensionField

func (m *extensionMap) Range(f func(protoreflect.FieldDescriptor, protoreflect.Value) bool) {
	if m != nil {
		for _, x := range *m {
			xd := x.Type().TypeDescriptor()
			v := x.Value()
			if xd.IsList() && v.List().Len() == 0 {
				continue
			}
			if !f(xd, v) {
				return
			}
		}
	}
}
func (m *extensionMap) Has(xt protoreflect.ExtensionType) (ok bool) {
	if m == nil {
		return false
	}
	xd := xt.TypeDescriptor()
	x, ok := (*m)[int32(xd.Number())]
	if !ok {
		return false
	}
	switch {
	case xd.IsList():
		return x.Value().List().Len() > 0
	case xd.IsMap():
		return x.Value().Map().Len() > 0
	case xd.Message() != nil:
		return x.Value().Message().IsValid()
	}
	return true
}
func (m *extensionMap) Clear(xt protoreflect.ExtensionType) {
	delete(*m, int32(xt.TypeDescriptor().Number()))
}
func (m *extensionMap) Get(xt protoreflect.ExtensionType) protoreflect.Value {
	xd := xt.TypeDescriptor()
	if m != nil {
		if x, ok := (*m)[int32(xd.Number())]; ok {
			return x.Value()
		}
	}
	return xt.Zero()
}
func (m *extensionMap) Set(xt protoreflect.ExtensionType, v protoreflect.Value) {
	xd := xt.TypeDescriptor()
	isValid := true
	switch {
	case !xt.IsValidValue(v):
		isValid = false
	case xd.IsList():
		isValid = v.List().IsValid()
	case xd.IsMap():
		isValid = v.Map().IsValid()
	case xd.Message() != nil:
		isValid = v.Message().IsValid()
	}
	if !isValid {
		panic(fmt.Sprintf("%v: assigning invalid value", xt.TypeDescriptor().FullName()))
	}

	if *m == nil {
		*m = make(map[int32]ExtensionField)
	}
	var x ExtensionField
	x.Set(xt, v)
	(*m)[int32(xd.Number())] = x
}
func (m *extensionMap) Mutable(xt protoreflect.ExtensionType) protoreflect.Value {
	xd := xt.TypeDescriptor()
	if xd.Kind() != protoreflect.MessageKind && xd.Kind() != protoreflect.GroupKind && !xd.IsList() && !xd.IsMap() {
		panic("invalid Mutable on field with non-composite type")
	}
	if x, ok := (*m)[int32(xd.Number())]; ok {
		return x.Value()
	}
	v := xt.New()
	m.Set(xt, v)
	return v
}

// MessageState is a data structure that is nested as the first field in a
// concrete message. It provides a way to implement the ProtoReflect method
// in an allocation-free way without needing to have a shadow Go type generated
// for every message type. This technique only works using unsafe.
//
// Example generated code:
//
//	type M struct {
//		state protoimpl.MessageState
//
//		Field1 int32
//		Field2 string
//		Field3 *BarMessage
//		...
//	}
//
//	func (m *M) ProtoReflect() protoreflect.Message {
//		mi := &file_fizz_buzz_proto_msgInfos[5]
//		if protoimpl.UnsafeEnabled && m != nil {
//			ms := protoimpl.X.MessageStateOf(Pointer(m))
//			if ms.LoadMessageInfo() == nil {
//				ms.StoreMessageInfo(mi)
//			}
//			return ms
//		}
//		return mi.MessageOf(m)
//	}
//
// The MessageState type holds a *MessageInfo, which must be atomically set to
// the message info associated with a given message instance.
// By unsafely converting a *M into a *MessageState, the MessageState object
// has access to all the information needed to implement protobuf reflection.
// It has access to the message info as its first field, and a pointer to the
// MessageState is identical to a pointer to the concrete message value.
//
// Requirements:
//   - The type M must implement protoreflect.ProtoMessage.
//   - The address of m must not be nil.
//   - The address of m and the address of m.state must be equal,
//     even though they are different Go types.
type MessageState struct {
	pragma.NoUnkeyedLiterals
	pragma.DoNotCompare
	pragma.DoNotCopy

	atomicMessageInfo *MessageInfo
}

type messageState MessageState

var (
	_ protoreflect.Message = (*messageState)(nil)
	_ unwrapper            = (*messageState)(nil)
)

// messageDataType is a tuple of a pointer to the message data and
// a pointer to the message type. It is a generalized way of providing a
// reflective view over a message instance. The disadvantage of this approach
// is the need to allocate this tuple of 16B.
type messageDataType struct {
	p  pointer
	mi *MessageInfo
}

type (
	messageReflectWrapper messageDataType
	messageIfaceWrapper   messageDataType
)

var (
	_ protoreflect.Message      = (*messageReflectWrapper)(nil)
	_ unwrapper                 = (*messageReflectWrapper)(nil)
	_ protoreflect.ProtoMessage = (*messageIfaceWrapper)(nil)
	_ unwrapper                 = (*messageIfaceWrapper)(nil)
)

// MessageOf returns a reflective view over a message. The input must be a
// pointer to a named Go struct. If the provided type has a ProtoReflect method,
// it must be implemented by calling this method.
func (mi *MessageInfo) MessageOf(m interface{}) protoreflect.Message {
	if reflect.TypeOf(m) != mi.GoReflectType {
		panic(fmt.Sprintf("type mismatch: got %T, want %v", m, mi.GoReflectType))
	}
	p := pointerOfIface(m)
	if p.IsNil() {
		return mi.nilMessage.Init(mi)
	}
	return &messageReflectWrapper{p, mi}
}

func (m *messageReflectWrapper) pointer() pointer          { return m.p }
func (m *messageReflectWrapper) messageInfo() *MessageInfo { return m.mi }

// Reset implements the v1 proto.Message.Reset method.
func (m *messageIfaceWrapper) Reset() {
	if mr, ok := m.protoUnwrap().(interface{ Reset() }); ok {
		mr.Reset()
		return
	}
	rv := reflect.ValueOf(m.protoUnwrap())
	if rv.Kind() == reflect.Ptr && !rv.IsNil() {
		rv.Elem().Set(reflect.Zero(rv.Type().Elem()))
	}
}
func (m *messageIfaceWrapper) ProtoReflect() protoreflect.Message {
	return (*messageReflectWrapper)(m)
}
func (m *messageIfaceWrapper) protoUnwrap() interface{} {
	return m.p.AsIfaceOf(m.mi.GoReflectType.Elem())
}

// checkField verifies that the provided field descriptor is valid.
// Exactly one of the returned values is populated.
func (mi *MessageInfo) checkField(fd protoreflect.FieldDescriptor) (*fieldInfo, protoreflect.ExtensionType) {
	var fi *fieldInfo
	if n := fd.Number(); 0 < n && int(n) < len(mi.denseFields) {
		fi = mi.denseFields[n]
	} else {
		fi = mi.fields[n]
	}
	if fi != nil {
		if fi.fieldDesc != fd {
			if got, want := fd.FullName(), fi.fieldDesc.FullName(); got != want {
				panic(fmt.Sprintf("mismatching field: got %v, want %v", got, want))
			}
			panic(fmt.Sprintf("mismatching field: %v", fd.FullName()))
		}
		return fi, nil
	}

	if fd.IsExtension() {
		if got, want := fd.ContainingMessage().FullName(), mi.Desc.FullName(); got != want {
			// TODO: Should this be exact containing message descriptor match?
			panic(fmt.Sprintf("extension %v has mismatching containing message: got %v, want %v", fd.FullName(), got, want))
		}
		if !mi.Desc.ExtensionRanges().Has(fd.Number()) {
			panic(fmt.Sprintf("extension %v extends %v outside the extension range", fd.FullName(), mi.Desc.FullName()))
		}
		xtd, ok := fd.(protoreflect.ExtensionTypeDescriptor)
		if !ok {
			panic(fmt.Sprintf("extension %v does not implement protoreflect.ExtensionTypeDescriptor", fd.FullName()))
		}
		return nil, xtd.Type()
	}
	panic(fmt.Sprintf("field %v is invalid", fd.FullName()))
}
