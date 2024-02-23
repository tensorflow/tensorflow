// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// pointerCoderFuncs is a set of pointer encoding functions.
type pointerCoderFuncs struct {
	mi        *MessageInfo
	size      func(p pointer, f *coderFieldInfo, opts marshalOptions) int
	marshal   func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error)
	unmarshal func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error)
	isInit    func(p pointer, f *coderFieldInfo) error
	merge     func(dst, src pointer, f *coderFieldInfo, opts mergeOptions)
}

// valueCoderFuncs is a set of protoreflect.Value encoding functions.
type valueCoderFuncs struct {
	size      func(v protoreflect.Value, tagsize int, opts marshalOptions) int
	marshal   func(b []byte, v protoreflect.Value, wiretag uint64, opts marshalOptions) ([]byte, error)
	unmarshal func(b []byte, v protoreflect.Value, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (protoreflect.Value, unmarshalOutput, error)
	isInit    func(v protoreflect.Value) error
	merge     func(dst, src protoreflect.Value, opts mergeOptions) protoreflect.Value
}

// fieldCoder returns pointer functions for a field, used for operating on
// struct fields.
func fieldCoder(fd protoreflect.FieldDescriptor, ft reflect.Type) (*MessageInfo, pointerCoderFuncs) {
	switch {
	case fd.IsMap():
		return encoderFuncsForMap(fd, ft)
	case fd.Cardinality() == protoreflect.Repeated && !fd.IsPacked():
		// Repeated fields (not packed).
		if ft.Kind() != reflect.Slice {
			break
		}
		ft := ft.Elem()
		switch fd.Kind() {
		case protoreflect.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolSlice
			}
		case protoreflect.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumSlice
			}
		case protoreflect.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32Slice
			}
		case protoreflect.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32Slice
			}
		case protoreflect.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32Slice
			}
		case protoreflect.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64Slice
			}
		case protoreflect.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64Slice
			}
		case protoreflect.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64Slice
			}
		case protoreflect.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32Slice
			}
		case protoreflect.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32Slice
			}
		case protoreflect.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatSlice
			}
		case protoreflect.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64Slice
			}
		case protoreflect.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64Slice
			}
		case protoreflect.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoubleSlice
			}
		case protoreflect.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringSliceValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringSlice
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesSliceValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesSlice
			}
		case protoreflect.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringSlice
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesSlice
			}
		case protoreflect.MessageKind:
			return getMessageInfo(ft), makeMessageSliceFieldCoder(fd, ft)
		case protoreflect.GroupKind:
			return getMessageInfo(ft), makeGroupSliceFieldCoder(fd, ft)
		}
	case fd.Cardinality() == protoreflect.Repeated && fd.IsPacked():
		// Packed repeated fields.
		//
		// Only repeated fields of primitive numeric types
		// (Varint, Fixed32, or Fixed64 wire type) can be packed.
		if ft.Kind() != reflect.Slice {
			break
		}
		ft := ft.Elem()
		switch fd.Kind() {
		case protoreflect.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolPackedSlice
			}
		case protoreflect.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumPackedSlice
			}
		case protoreflect.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32PackedSlice
			}
		case protoreflect.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32PackedSlice
			}
		case protoreflect.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32PackedSlice
			}
		case protoreflect.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64PackedSlice
			}
		case protoreflect.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64PackedSlice
			}
		case protoreflect.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64PackedSlice
			}
		case protoreflect.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32PackedSlice
			}
		case protoreflect.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32PackedSlice
			}
		case protoreflect.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatPackedSlice
			}
		case protoreflect.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64PackedSlice
			}
		case protoreflect.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64PackedSlice
			}
		case protoreflect.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoublePackedSlice
			}
		}
	case fd.Kind() == protoreflect.MessageKind:
		return getMessageInfo(ft), makeMessageFieldCoder(fd, ft)
	case fd.Kind() == protoreflect.GroupKind:
		return getMessageInfo(ft), makeGroupFieldCoder(fd, ft)
	case !fd.HasPresence() && fd.ContainingOneof() == nil:
		// Populated oneof fields always encode even if set to the zero value,
		// which normally are not encoded in proto3.
		switch fd.Kind() {
		case protoreflect.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolNoZero
			}
		case protoreflect.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumNoZero
			}
		case protoreflect.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32NoZero
			}
		case protoreflect.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32NoZero
			}
		case protoreflect.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32NoZero
			}
		case protoreflect.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64NoZero
			}
		case protoreflect.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64NoZero
			}
		case protoreflect.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64NoZero
			}
		case protoreflect.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32NoZero
			}
		case protoreflect.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32NoZero
			}
		case protoreflect.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatNoZero
			}
		case protoreflect.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64NoZero
			}
		case protoreflect.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64NoZero
			}
		case protoreflect.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoubleNoZero
			}
		case protoreflect.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringNoZeroValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringNoZero
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesNoZeroValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesNoZero
			}
		case protoreflect.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringNoZero
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytesNoZero
			}
		}
	case ft.Kind() == reflect.Ptr:
		ft := ft.Elem()
		switch fd.Kind() {
		case protoreflect.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBoolPtr
			}
		case protoreflect.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnumPtr
			}
		case protoreflect.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32Ptr
			}
		case protoreflect.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32Ptr
			}
		case protoreflect.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32Ptr
			}
		case protoreflect.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64Ptr
			}
		case protoreflect.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64Ptr
			}
		case protoreflect.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64Ptr
			}
		case protoreflect.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32Ptr
			}
		case protoreflect.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32Ptr
			}
		case protoreflect.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloatPtr
			}
		case protoreflect.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64Ptr
			}
		case protoreflect.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64Ptr
			}
		case protoreflect.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDoublePtr
			}
		case protoreflect.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringPtrValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderStringPtr
			}
		case protoreflect.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderStringPtr
			}
		}
	default:
		switch fd.Kind() {
		case protoreflect.BoolKind:
			if ft.Kind() == reflect.Bool {
				return nil, coderBool
			}
		case protoreflect.EnumKind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderEnum
			}
		case protoreflect.Int32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderInt32
			}
		case protoreflect.Sint32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSint32
			}
		case protoreflect.Uint32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderUint32
			}
		case protoreflect.Int64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderInt64
			}
		case protoreflect.Sint64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSint64
			}
		case protoreflect.Uint64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderUint64
			}
		case protoreflect.Sfixed32Kind:
			if ft.Kind() == reflect.Int32 {
				return nil, coderSfixed32
			}
		case protoreflect.Fixed32Kind:
			if ft.Kind() == reflect.Uint32 {
				return nil, coderFixed32
			}
		case protoreflect.FloatKind:
			if ft.Kind() == reflect.Float32 {
				return nil, coderFloat
			}
		case protoreflect.Sfixed64Kind:
			if ft.Kind() == reflect.Int64 {
				return nil, coderSfixed64
			}
		case protoreflect.Fixed64Kind:
			if ft.Kind() == reflect.Uint64 {
				return nil, coderFixed64
			}
		case protoreflect.DoubleKind:
			if ft.Kind() == reflect.Float64 {
				return nil, coderDouble
			}
		case protoreflect.StringKind:
			if ft.Kind() == reflect.String && strs.EnforceUTF8(fd) {
				return nil, coderStringValidateUTF8
			}
			if ft.Kind() == reflect.String {
				return nil, coderString
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 && strs.EnforceUTF8(fd) {
				return nil, coderBytesValidateUTF8
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytes
			}
		case protoreflect.BytesKind:
			if ft.Kind() == reflect.String {
				return nil, coderString
			}
			if ft.Kind() == reflect.Slice && ft.Elem().Kind() == reflect.Uint8 {
				return nil, coderBytes
			}
		}
	}
	panic(fmt.Sprintf("invalid type: no encoder for %v %v %v/%v", fd.FullName(), fd.Cardinality(), fd.Kind(), ft))
}

// encoderFuncsForValue returns value functions for a field, used for
// extension values and map encoding.
func encoderFuncsForValue(fd protoreflect.FieldDescriptor) valueCoderFuncs {
	switch {
	case fd.Cardinality() == protoreflect.Repeated && !fd.IsPacked():
		switch fd.Kind() {
		case protoreflect.BoolKind:
			return coderBoolSliceValue
		case protoreflect.EnumKind:
			return coderEnumSliceValue
		case protoreflect.Int32Kind:
			return coderInt32SliceValue
		case protoreflect.Sint32Kind:
			return coderSint32SliceValue
		case protoreflect.Uint32Kind:
			return coderUint32SliceValue
		case protoreflect.Int64Kind:
			return coderInt64SliceValue
		case protoreflect.Sint64Kind:
			return coderSint64SliceValue
		case protoreflect.Uint64Kind:
			return coderUint64SliceValue
		case protoreflect.Sfixed32Kind:
			return coderSfixed32SliceValue
		case protoreflect.Fixed32Kind:
			return coderFixed32SliceValue
		case protoreflect.FloatKind:
			return coderFloatSliceValue
		case protoreflect.Sfixed64Kind:
			return coderSfixed64SliceValue
		case protoreflect.Fixed64Kind:
			return coderFixed64SliceValue
		case protoreflect.DoubleKind:
			return coderDoubleSliceValue
		case protoreflect.StringKind:
			// We don't have a UTF-8 validating coder for repeated string fields.
			// Value coders are used for extensions and maps.
			// Extensions are never proto3, and maps never contain lists.
			return coderStringSliceValue
		case protoreflect.BytesKind:
			return coderBytesSliceValue
		case protoreflect.MessageKind:
			return coderMessageSliceValue
		case protoreflect.GroupKind:
			return coderGroupSliceValue
		}
	case fd.Cardinality() == protoreflect.Repeated && fd.IsPacked():
		switch fd.Kind() {
		case protoreflect.BoolKind:
			return coderBoolPackedSliceValue
		case protoreflect.EnumKind:
			return coderEnumPackedSliceValue
		case protoreflect.Int32Kind:
			return coderInt32PackedSliceValue
		case protoreflect.Sint32Kind:
			return coderSint32PackedSliceValue
		case protoreflect.Uint32Kind:
			return coderUint32PackedSliceValue
		case protoreflect.Int64Kind:
			return coderInt64PackedSliceValue
		case protoreflect.Sint64Kind:
			return coderSint64PackedSliceValue
		case protoreflect.Uint64Kind:
			return coderUint64PackedSliceValue
		case protoreflect.Sfixed32Kind:
			return coderSfixed32PackedSliceValue
		case protoreflect.Fixed32Kind:
			return coderFixed32PackedSliceValue
		case protoreflect.FloatKind:
			return coderFloatPackedSliceValue
		case protoreflect.Sfixed64Kind:
			return coderSfixed64PackedSliceValue
		case protoreflect.Fixed64Kind:
			return coderFixed64PackedSliceValue
		case protoreflect.DoubleKind:
			return coderDoublePackedSliceValue
		}
	default:
		switch fd.Kind() {
		default:
		case protoreflect.BoolKind:
			return coderBoolValue
		case protoreflect.EnumKind:
			return coderEnumValue
		case protoreflect.Int32Kind:
			return coderInt32Value
		case protoreflect.Sint32Kind:
			return coderSint32Value
		case protoreflect.Uint32Kind:
			return coderUint32Value
		case protoreflect.Int64Kind:
			return coderInt64Value
		case protoreflect.Sint64Kind:
			return coderSint64Value
		case protoreflect.Uint64Kind:
			return coderUint64Value
		case protoreflect.Sfixed32Kind:
			return coderSfixed32Value
		case protoreflect.Fixed32Kind:
			return coderFixed32Value
		case protoreflect.FloatKind:
			return coderFloatValue
		case protoreflect.Sfixed64Kind:
			return coderSfixed64Value
		case protoreflect.Fixed64Kind:
			return coderFixed64Value
		case protoreflect.DoubleKind:
			return coderDoubleValue
		case protoreflect.StringKind:
			if strs.EnforceUTF8(fd) {
				return coderStringValueValidateUTF8
			}
			return coderStringValue
		case protoreflect.BytesKind:
			return coderBytesValue
		case protoreflect.MessageKind:
			return coderMessageValue
		case protoreflect.GroupKind:
			return coderGroupValue
		}
	}
	panic(fmt.Sprintf("invalid field: no encoder for %v %v %v", fd.FullName(), fd.Cardinality(), fd.Kind()))
}
