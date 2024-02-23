// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build purego || appengine
// +build purego appengine

package impl

import (
	"fmt"
	"reflect"
	"sync"
)

const UnsafeEnabled = false

// Pointer is an opaque pointer type.
type Pointer interface{}

// offset represents the offset to a struct field, accessible from a pointer.
// The offset is the field index into a struct.
type offset struct {
	index  int
	export exporter
}

// offsetOf returns a field offset for the struct field.
func offsetOf(f reflect.StructField, x exporter) offset {
	if len(f.Index) != 1 {
		panic("embedded structs are not supported")
	}
	if f.PkgPath == "" {
		return offset{index: f.Index[0]} // field is already exported
	}
	if x == nil {
		panic("exporter must be provided for unexported field")
	}
	return offset{index: f.Index[0], export: x}
}

// IsValid reports whether the offset is valid.
func (f offset) IsValid() bool { return f.index >= 0 }

// invalidOffset is an invalid field offset.
var invalidOffset = offset{index: -1}

// zeroOffset is a noop when calling pointer.Apply.
var zeroOffset = offset{index: 0}

// pointer is an abstract representation of a pointer to a struct or field.
type pointer struct{ v reflect.Value }

// pointerOf returns p as a pointer.
func pointerOf(p Pointer) pointer {
	return pointerOfIface(p)
}

// pointerOfValue returns v as a pointer.
func pointerOfValue(v reflect.Value) pointer {
	return pointer{v: v}
}

// pointerOfIface returns the pointer portion of an interface.
func pointerOfIface(v interface{}) pointer {
	return pointer{v: reflect.ValueOf(v)}
}

// IsNil reports whether the pointer is nil.
func (p pointer) IsNil() bool {
	return p.v.IsNil()
}

// Apply adds an offset to the pointer to derive a new pointer
// to a specified field. The current pointer must be pointing at a struct.
func (p pointer) Apply(f offset) pointer {
	if f.export != nil {
		if v := reflect.ValueOf(f.export(p.v.Interface(), f.index)); v.IsValid() {
			return pointer{v: v}
		}
	}
	return pointer{v: p.v.Elem().Field(f.index).Addr()}
}

// AsValueOf treats p as a pointer to an object of type t and returns the value.
// It is equivalent to reflect.ValueOf(p.AsIfaceOf(t))
func (p pointer) AsValueOf(t reflect.Type) reflect.Value {
	if got := p.v.Type().Elem(); got != t {
		panic(fmt.Sprintf("invalid type: got %v, want %v", got, t))
	}
	return p.v
}

// AsIfaceOf treats p as a pointer to an object of type t and returns the value.
// It is equivalent to p.AsValueOf(t).Interface()
func (p pointer) AsIfaceOf(t reflect.Type) interface{} {
	return p.AsValueOf(t).Interface()
}

func (p pointer) Bool() *bool              { return p.v.Interface().(*bool) }
func (p pointer) BoolPtr() **bool          { return p.v.Interface().(**bool) }
func (p pointer) BoolSlice() *[]bool       { return p.v.Interface().(*[]bool) }
func (p pointer) Int32() *int32            { return p.v.Interface().(*int32) }
func (p pointer) Int32Ptr() **int32        { return p.v.Interface().(**int32) }
func (p pointer) Int32Slice() *[]int32     { return p.v.Interface().(*[]int32) }
func (p pointer) Int64() *int64            { return p.v.Interface().(*int64) }
func (p pointer) Int64Ptr() **int64        { return p.v.Interface().(**int64) }
func (p pointer) Int64Slice() *[]int64     { return p.v.Interface().(*[]int64) }
func (p pointer) Uint32() *uint32          { return p.v.Interface().(*uint32) }
func (p pointer) Uint32Ptr() **uint32      { return p.v.Interface().(**uint32) }
func (p pointer) Uint32Slice() *[]uint32   { return p.v.Interface().(*[]uint32) }
func (p pointer) Uint64() *uint64          { return p.v.Interface().(*uint64) }
func (p pointer) Uint64Ptr() **uint64      { return p.v.Interface().(**uint64) }
func (p pointer) Uint64Slice() *[]uint64   { return p.v.Interface().(*[]uint64) }
func (p pointer) Float32() *float32        { return p.v.Interface().(*float32) }
func (p pointer) Float32Ptr() **float32    { return p.v.Interface().(**float32) }
func (p pointer) Float32Slice() *[]float32 { return p.v.Interface().(*[]float32) }
func (p pointer) Float64() *float64        { return p.v.Interface().(*float64) }
func (p pointer) Float64Ptr() **float64    { return p.v.Interface().(**float64) }
func (p pointer) Float64Slice() *[]float64 { return p.v.Interface().(*[]float64) }
func (p pointer) String() *string          { return p.v.Interface().(*string) }
func (p pointer) StringPtr() **string      { return p.v.Interface().(**string) }
func (p pointer) StringSlice() *[]string   { return p.v.Interface().(*[]string) }
func (p pointer) Bytes() *[]byte           { return p.v.Interface().(*[]byte) }
func (p pointer) BytesPtr() **[]byte       { return p.v.Interface().(**[]byte) }
func (p pointer) BytesSlice() *[][]byte    { return p.v.Interface().(*[][]byte) }
func (p pointer) WeakFields() *weakFields  { return (*weakFields)(p.v.Interface().(*WeakFields)) }
func (p pointer) Extensions() *map[int32]ExtensionField {
	return p.v.Interface().(*map[int32]ExtensionField)
}

func (p pointer) Elem() pointer {
	return pointer{v: p.v.Elem()}
}

// PointerSlice copies []*T from p as a new []pointer.
// This behavior differs from the implementation in pointer_unsafe.go.
func (p pointer) PointerSlice() []pointer {
	// TODO: reconsider this
	if p.v.IsNil() {
		return nil
	}
	n := p.v.Elem().Len()
	s := make([]pointer, n)
	for i := 0; i < n; i++ {
		s[i] = pointer{v: p.v.Elem().Index(i)}
	}
	return s
}

// AppendPointerSlice appends v to p, which must be a []*T.
func (p pointer) AppendPointerSlice(v pointer) {
	sp := p.v.Elem()
	sp.Set(reflect.Append(sp, v.v))
}

// SetPointer sets *p to v.
func (p pointer) SetPointer(v pointer) {
	p.v.Elem().Set(v.v)
}

func growSlice(p pointer, addCap int) {
	// TODO: Once we only support Go 1.20 and newer, use reflect.Grow.
	in := p.v.Elem()
	out := reflect.MakeSlice(in.Type(), in.Len(), in.Len()+addCap)
	reflect.Copy(out, in)
	p.v.Elem().Set(out)
}

func (p pointer) growBoolSlice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growInt32Slice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growUint32Slice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growInt64Slice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growUint64Slice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growFloat64Slice(addCap int) {
	growSlice(p, addCap)
}

func (p pointer) growFloat32Slice(addCap int) {
	growSlice(p, addCap)
}

func (Export) MessageStateOf(p Pointer) *messageState     { panic("not supported") }
func (ms *messageState) pointer() pointer                 { panic("not supported") }
func (ms *messageState) messageInfo() *MessageInfo        { panic("not supported") }
func (ms *messageState) LoadMessageInfo() *MessageInfo    { panic("not supported") }
func (ms *messageState) StoreMessageInfo(mi *MessageInfo) { panic("not supported") }

type atomicNilMessage struct {
	once sync.Once
	m    messageReflectWrapper
}

func (m *atomicNilMessage) Init(mi *MessageInfo) *messageReflectWrapper {
	m.once.Do(func() {
		m.m.p = pointerOfIface(reflect.Zero(mi.GoReflectType).Interface())
		m.m.mi = mi
	})
	return &m.m
}
