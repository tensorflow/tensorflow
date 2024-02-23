// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"math"
	"reflect"
	"sync"

	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

type fieldInfo struct {
	fieldDesc protoreflect.FieldDescriptor

	// These fields are used for protobuf reflection support.
	has        func(pointer) bool
	clear      func(pointer)
	get        func(pointer) protoreflect.Value
	set        func(pointer, protoreflect.Value)
	mutable    func(pointer) protoreflect.Value
	newMessage func() protoreflect.Message
	newField   func() protoreflect.Value
}

func fieldInfoForMissing(fd protoreflect.FieldDescriptor) fieldInfo {
	// This never occurs for generated message types.
	// It implies that a hand-crafted type has missing Go fields
	// for specific protobuf message fields.
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			return false
		},
		clear: func(p pointer) {
			panic("missing Go struct field for " + string(fd.FullName()))
		},
		get: func(p pointer) protoreflect.Value {
			return fd.Default()
		},
		set: func(p pointer, v protoreflect.Value) {
			panic("missing Go struct field for " + string(fd.FullName()))
		},
		mutable: func(p pointer) protoreflect.Value {
			panic("missing Go struct field for " + string(fd.FullName()))
		},
		newMessage: func() protoreflect.Message {
			panic("missing Go struct field for " + string(fd.FullName()))
		},
		newField: func() protoreflect.Value {
			if v := fd.Default(); v.IsValid() {
				return v
			}
			panic("missing Go struct field for " + string(fd.FullName()))
		},
	}
}

func fieldInfoForOneof(fd protoreflect.FieldDescriptor, fs reflect.StructField, x exporter, ot reflect.Type) fieldInfo {
	ft := fs.Type
	if ft.Kind() != reflect.Interface {
		panic(fmt.Sprintf("field %v has invalid type: got %v, want interface kind", fd.FullName(), ft))
	}
	if ot.Kind() != reflect.Struct {
		panic(fmt.Sprintf("field %v has invalid type: got %v, want struct kind", fd.FullName(), ot))
	}
	if !reflect.PtrTo(ot).Implements(ft) {
		panic(fmt.Sprintf("field %v has invalid type: %v does not implement %v", fd.FullName(), ot, ft))
	}
	conv := NewConverter(ot.Field(0).Type, fd)
	isMessage := fd.Message() != nil

	// TODO: Implement unsafe fast path?
	fieldOffset := offsetOf(fs, x)
	return fieldInfo{
		// NOTE: The logic below intentionally assumes that oneof fields are
		// well-formatted. That is, the oneof interface never contains a
		// typed nil pointer to one of the wrapper structs.

		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() || rv.Elem().Type().Elem() != ot || rv.Elem().IsNil() {
				return false
			}
			return true
		},
		clear: func(p pointer) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() || rv.Elem().Type().Elem() != ot {
				// NOTE: We intentionally don't check for rv.Elem().IsNil()
				// so that (*OneofWrapperType)(nil) gets cleared to nil.
				return
			}
			rv.Set(reflect.Zero(rv.Type()))
		},
		get: func(p pointer) protoreflect.Value {
			if p.IsNil() {
				return conv.Zero()
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() || rv.Elem().Type().Elem() != ot || rv.Elem().IsNil() {
				return conv.Zero()
			}
			rv = rv.Elem().Elem().Field(0)
			return conv.PBValueOf(rv)
		},
		set: func(p pointer, v protoreflect.Value) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() || rv.Elem().Type().Elem() != ot || rv.Elem().IsNil() {
				rv.Set(reflect.New(ot))
			}
			rv = rv.Elem().Elem().Field(0)
			rv.Set(conv.GoValueOf(v))
		},
		mutable: func(p pointer) protoreflect.Value {
			if !isMessage {
				panic(fmt.Sprintf("field %v with invalid Mutable call on field with non-composite type", fd.FullName()))
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() || rv.Elem().Type().Elem() != ot || rv.Elem().IsNil() {
				rv.Set(reflect.New(ot))
			}
			rv = rv.Elem().Elem().Field(0)
			if rv.Kind() == reflect.Ptr && rv.IsNil() {
				rv.Set(conv.GoValueOf(protoreflect.ValueOfMessage(conv.New().Message())))
			}
			return conv.PBValueOf(rv)
		},
		newMessage: func() protoreflect.Message {
			return conv.New().Message()
		},
		newField: func() protoreflect.Value {
			return conv.New()
		},
	}
}

func fieldInfoForMap(fd protoreflect.FieldDescriptor, fs reflect.StructField, x exporter) fieldInfo {
	ft := fs.Type
	if ft.Kind() != reflect.Map {
		panic(fmt.Sprintf("field %v has invalid type: got %v, want map kind", fd.FullName(), ft))
	}
	conv := NewConverter(ft, fd)

	// TODO: Implement unsafe fast path?
	fieldOffset := offsetOf(fs, x)
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			return rv.Len() > 0
		},
		clear: func(p pointer) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			rv.Set(reflect.Zero(rv.Type()))
		},
		get: func(p pointer) protoreflect.Value {
			if p.IsNil() {
				return conv.Zero()
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.Len() == 0 {
				return conv.Zero()
			}
			return conv.PBValueOf(rv)
		},
		set: func(p pointer, v protoreflect.Value) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			pv := conv.GoValueOf(v)
			if pv.IsNil() {
				panic(fmt.Sprintf("map field %v cannot be set with read-only value", fd.FullName()))
			}
			rv.Set(pv)
		},
		mutable: func(p pointer) protoreflect.Value {
			v := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if v.IsNil() {
				v.Set(reflect.MakeMap(fs.Type))
			}
			return conv.PBValueOf(v)
		},
		newField: func() protoreflect.Value {
			return conv.New()
		},
	}
}

func fieldInfoForList(fd protoreflect.FieldDescriptor, fs reflect.StructField, x exporter) fieldInfo {
	ft := fs.Type
	if ft.Kind() != reflect.Slice {
		panic(fmt.Sprintf("field %v has invalid type: got %v, want slice kind", fd.FullName(), ft))
	}
	conv := NewConverter(reflect.PtrTo(ft), fd)

	// TODO: Implement unsafe fast path?
	fieldOffset := offsetOf(fs, x)
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			return rv.Len() > 0
		},
		clear: func(p pointer) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			rv.Set(reflect.Zero(rv.Type()))
		},
		get: func(p pointer) protoreflect.Value {
			if p.IsNil() {
				return conv.Zero()
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type)
			if rv.Elem().Len() == 0 {
				return conv.Zero()
			}
			return conv.PBValueOf(rv)
		},
		set: func(p pointer, v protoreflect.Value) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			pv := conv.GoValueOf(v)
			if pv.IsNil() {
				panic(fmt.Sprintf("list field %v cannot be set with read-only value", fd.FullName()))
			}
			rv.Set(pv.Elem())
		},
		mutable: func(p pointer) protoreflect.Value {
			v := p.Apply(fieldOffset).AsValueOf(fs.Type)
			return conv.PBValueOf(v)
		},
		newField: func() protoreflect.Value {
			return conv.New()
		},
	}
}

var (
	nilBytes   = reflect.ValueOf([]byte(nil))
	emptyBytes = reflect.ValueOf([]byte{})
)

func fieldInfoForScalar(fd protoreflect.FieldDescriptor, fs reflect.StructField, x exporter) fieldInfo {
	ft := fs.Type
	nullable := fd.HasPresence()
	isBytes := ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8
	if nullable {
		if ft.Kind() != reflect.Ptr && ft.Kind() != reflect.Slice {
			// This never occurs for generated message types.
			// Despite the protobuf type system specifying presence,
			// the Go field type cannot represent it.
			nullable = false
		}
		if ft.Kind() == reflect.Ptr {
			ft = ft.Elem()
		}
	}
	conv := NewConverter(ft, fd)

	// TODO: Implement unsafe fast path?
	fieldOffset := offsetOf(fs, x)
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if nullable {
				return !rv.IsNil()
			}
			switch rv.Kind() {
			case reflect.Bool:
				return rv.Bool()
			case reflect.Int32, reflect.Int64:
				return rv.Int() != 0
			case reflect.Uint32, reflect.Uint64:
				return rv.Uint() != 0
			case reflect.Float32, reflect.Float64:
				return rv.Float() != 0 || math.Signbit(rv.Float())
			case reflect.String, reflect.Slice:
				return rv.Len() > 0
			default:
				panic(fmt.Sprintf("field %v has invalid type: %v", fd.FullName(), rv.Type())) // should never happen
			}
		},
		clear: func(p pointer) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			rv.Set(reflect.Zero(rv.Type()))
		},
		get: func(p pointer) protoreflect.Value {
			if p.IsNil() {
				return conv.Zero()
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if nullable {
				if rv.IsNil() {
					return conv.Zero()
				}
				if rv.Kind() == reflect.Ptr {
					rv = rv.Elem()
				}
			}
			return conv.PBValueOf(rv)
		},
		set: func(p pointer, v protoreflect.Value) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if nullable && rv.Kind() == reflect.Ptr {
				if rv.IsNil() {
					rv.Set(reflect.New(ft))
				}
				rv = rv.Elem()
			}
			rv.Set(conv.GoValueOf(v))
			if isBytes && rv.Len() == 0 {
				if nullable {
					rv.Set(emptyBytes) // preserve presence
				} else {
					rv.Set(nilBytes) // do not preserve presence
				}
			}
		},
		newField: func() protoreflect.Value {
			return conv.New()
		},
	}
}

func fieldInfoForWeakMessage(fd protoreflect.FieldDescriptor, weakOffset offset) fieldInfo {
	if !flags.ProtoLegacy {
		panic("no support for proto1 weak fields")
	}

	var once sync.Once
	var messageType protoreflect.MessageType
	lazyInit := func() {
		once.Do(func() {
			messageName := fd.Message().FullName()
			messageType, _ = protoregistry.GlobalTypes.FindMessageByName(messageName)
			if messageType == nil {
				panic(fmt.Sprintf("weak message %v for field %v is not linked in", messageName, fd.FullName()))
			}
		})
	}

	num := fd.Number()
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			_, ok := p.Apply(weakOffset).WeakFields().get(num)
			return ok
		},
		clear: func(p pointer) {
			p.Apply(weakOffset).WeakFields().clear(num)
		},
		get: func(p pointer) protoreflect.Value {
			lazyInit()
			if p.IsNil() {
				return protoreflect.ValueOfMessage(messageType.Zero())
			}
			m, ok := p.Apply(weakOffset).WeakFields().get(num)
			if !ok {
				return protoreflect.ValueOfMessage(messageType.Zero())
			}
			return protoreflect.ValueOfMessage(m.ProtoReflect())
		},
		set: func(p pointer, v protoreflect.Value) {
			lazyInit()
			m := v.Message()
			if m.Descriptor() != messageType.Descriptor() {
				if got, want := m.Descriptor().FullName(), messageType.Descriptor().FullName(); got != want {
					panic(fmt.Sprintf("field %v has mismatching message descriptor: got %v, want %v", fd.FullName(), got, want))
				}
				panic(fmt.Sprintf("field %v has mismatching message descriptor: %v", fd.FullName(), m.Descriptor().FullName()))
			}
			p.Apply(weakOffset).WeakFields().set(num, m.Interface())
		},
		mutable: func(p pointer) protoreflect.Value {
			lazyInit()
			fs := p.Apply(weakOffset).WeakFields()
			m, ok := fs.get(num)
			if !ok {
				m = messageType.New().Interface()
				fs.set(num, m)
			}
			return protoreflect.ValueOfMessage(m.ProtoReflect())
		},
		newMessage: func() protoreflect.Message {
			lazyInit()
			return messageType.New()
		},
		newField: func() protoreflect.Value {
			lazyInit()
			return protoreflect.ValueOfMessage(messageType.New())
		},
	}
}

func fieldInfoForMessage(fd protoreflect.FieldDescriptor, fs reflect.StructField, x exporter) fieldInfo {
	ft := fs.Type
	conv := NewConverter(ft, fd)

	// TODO: Implement unsafe fast path?
	fieldOffset := offsetOf(fs, x)
	return fieldInfo{
		fieldDesc: fd,
		has: func(p pointer) bool {
			if p.IsNil() {
				return false
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if fs.Type.Kind() != reflect.Ptr {
				return !isZero(rv)
			}
			return !rv.IsNil()
		},
		clear: func(p pointer) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			rv.Set(reflect.Zero(rv.Type()))
		},
		get: func(p pointer) protoreflect.Value {
			if p.IsNil() {
				return conv.Zero()
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			return conv.PBValueOf(rv)
		},
		set: func(p pointer, v protoreflect.Value) {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			rv.Set(conv.GoValueOf(v))
			if fs.Type.Kind() == reflect.Ptr && rv.IsNil() {
				panic(fmt.Sprintf("field %v has invalid nil pointer", fd.FullName()))
			}
		},
		mutable: func(p pointer) protoreflect.Value {
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if fs.Type.Kind() == reflect.Ptr && rv.IsNil() {
				rv.Set(conv.GoValueOf(conv.New()))
			}
			return conv.PBValueOf(rv)
		},
		newMessage: func() protoreflect.Message {
			return conv.New().Message()
		},
		newField: func() protoreflect.Value {
			return conv.New()
		},
	}
}

type oneofInfo struct {
	oneofDesc protoreflect.OneofDescriptor
	which     func(pointer) protoreflect.FieldNumber
}

func makeOneofInfo(od protoreflect.OneofDescriptor, si structInfo, x exporter) *oneofInfo {
	oi := &oneofInfo{oneofDesc: od}
	if od.IsSynthetic() {
		fs := si.fieldsByNumber[od.Fields().Get(0).Number()]
		fieldOffset := offsetOf(fs, x)
		oi.which = func(p pointer) protoreflect.FieldNumber {
			if p.IsNil() {
				return 0
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() { // valid on either *T or []byte
				return 0
			}
			return od.Fields().Get(0).Number()
		}
	} else {
		fs := si.oneofsByName[od.Name()]
		fieldOffset := offsetOf(fs, x)
		oi.which = func(p pointer) protoreflect.FieldNumber {
			if p.IsNil() {
				return 0
			}
			rv := p.Apply(fieldOffset).AsValueOf(fs.Type).Elem()
			if rv.IsNil() {
				return 0
			}
			rv = rv.Elem()
			if rv.IsNil() {
				return 0
			}
			return si.oneofWrappersByType[rv.Type().Elem()]
		}
	}
	return oi
}

// isZero is identical to reflect.Value.IsZero.
// TODO: Remove this when Go1.13 is the minimally supported Go version.
func isZero(v reflect.Value) bool {
	switch v.Kind() {
	case reflect.Bool:
		return !v.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return v.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return math.Float64bits(v.Float()) == 0
	case reflect.Complex64, reflect.Complex128:
		c := v.Complex()
		return math.Float64bits(real(c)) == 0 && math.Float64bits(imag(c)) == 0
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			if !isZero(v.Index(i)) {
				return false
			}
		}
		return true
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice, reflect.UnsafePointer:
		return v.IsNil()
	case reflect.String:
		return v.Len() == 0
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			if !isZero(v.Field(i)) {
				return false
			}
		}
		return true
	default:
		panic(&reflect.ValueError{Method: "reflect.Value.IsZero", Kind: v.Kind()})
	}
}
