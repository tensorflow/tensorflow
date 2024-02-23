// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/reflect/protoreflect"
)

// HasExtension reports whether an extension field is populated.
// It returns false if m is invalid or if xt does not extend m.
func HasExtension(m Message, xt protoreflect.ExtensionType) bool {
	// Treat nil message interface as an empty message; no populated fields.
	if m == nil {
		return false
	}

	// As a special-case, we reports invalid or mismatching descriptors
	// as always not being populated (since they aren't).
	if xt == nil || m.ProtoReflect().Descriptor() != xt.TypeDescriptor().ContainingMessage() {
		return false
	}

	return m.ProtoReflect().Has(xt.TypeDescriptor())
}

// ClearExtension clears an extension field such that subsequent
// [HasExtension] calls return false.
// It panics if m is invalid or if xt does not extend m.
func ClearExtension(m Message, xt protoreflect.ExtensionType) {
	m.ProtoReflect().Clear(xt.TypeDescriptor())
}

// GetExtension retrieves the value for an extension field.
// If the field is unpopulated, it returns the default value for
// scalars and an immutable, empty value for lists or messages.
// It panics if xt does not extend m.
func GetExtension(m Message, xt protoreflect.ExtensionType) interface{} {
	// Treat nil message interface as an empty message; return the default.
	if m == nil {
		return xt.InterfaceOf(xt.Zero())
	}

	return xt.InterfaceOf(m.ProtoReflect().Get(xt.TypeDescriptor()))
}

// SetExtension stores the value of an extension field.
// It panics if m is invalid, xt does not extend m, or if type of v
// is invalid for the specified extension field.
func SetExtension(m Message, xt protoreflect.ExtensionType, v interface{}) {
	xd := xt.TypeDescriptor()
	pv := xt.ValueOf(v)

	// Specially treat an invalid list, map, or message as clear.
	isValid := true
	switch {
	case xd.IsList():
		isValid = pv.List().IsValid()
	case xd.IsMap():
		isValid = pv.Map().IsValid()
	case xd.Message() != nil:
		isValid = pv.Message().IsValid()
	}
	if !isValid {
		m.ProtoReflect().Clear(xd)
		return
	}

	m.ProtoReflect().Set(xd, pv)
}

// RangeExtensions iterates over every populated extension field in m in an
// undefined order, calling f for each extension type and value encountered.
// It returns immediately if f returns false.
// While iterating, mutating operations may only be performed
// on the current extension field.
func RangeExtensions(m Message, f func(protoreflect.ExtensionType, interface{}) bool) {
	// Treat nil message interface as an empty message; nothing to range over.
	if m == nil {
		return
	}

	m.ProtoReflect().Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		if fd.IsExtension() {
			xt := fd.(protoreflect.ExtensionTypeDescriptor).Type()
			vi := xt.InterfaceOf(v)
			return f(xt, vi)
		}
		return true
	})
}
