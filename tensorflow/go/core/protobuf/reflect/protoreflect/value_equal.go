// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import (
	"bytes"
	"fmt"
	"math"
	"reflect"

	"google.golang.org/protobuf/encoding/protowire"
)

// Equal reports whether v1 and v2 are recursively equal.
//
//   - Values of different types are always unequal.
//
//   - Bytes values are equal if they contain identical bytes.
//     Empty bytes (regardless of nil-ness) are considered equal.
//
//   - Floating point values are equal if they contain the same value.
//     Unlike the == operator, a NaN is equal to another NaN.
//
//   - Enums are equal if they contain the same number.
//     Since [Value] does not contain an enum descriptor,
//     enum values do not consider the type of the enum.
//
//   - Other scalar values are equal if they contain the same value.
//
//   - [Message] values are equal if they belong to the same message descriptor,
//     have the same set of populated known and extension field values,
//     and the same set of unknown fields values.
//
//   - [List] values are equal if they are the same length and
//     each corresponding element is equal.
//
//   - [Map] values are equal if they have the same set of keys and
//     the corresponding value for each key is equal.
func (v1 Value) Equal(v2 Value) bool {
	return equalValue(v1, v2)
}

func equalValue(x, y Value) bool {
	eqType := x.typ == y.typ
	switch x.typ {
	case nilType:
		return eqType
	case boolType:
		return eqType && x.Bool() == y.Bool()
	case int32Type, int64Type:
		return eqType && x.Int() == y.Int()
	case uint32Type, uint64Type:
		return eqType && x.Uint() == y.Uint()
	case float32Type, float64Type:
		return eqType && equalFloat(x.Float(), y.Float())
	case stringType:
		return eqType && x.String() == y.String()
	case bytesType:
		return eqType && bytes.Equal(x.Bytes(), y.Bytes())
	case enumType:
		return eqType && x.Enum() == y.Enum()
	default:
		switch x := x.Interface().(type) {
		case Message:
			y, ok := y.Interface().(Message)
			return ok && equalMessage(x, y)
		case List:
			y, ok := y.Interface().(List)
			return ok && equalList(x, y)
		case Map:
			y, ok := y.Interface().(Map)
			return ok && equalMap(x, y)
		default:
			panic(fmt.Sprintf("unknown type: %T", x))
		}
	}
}

// equalFloat compares two floats, where NaNs are treated as equal.
func equalFloat(x, y float64) bool {
	if math.IsNaN(x) || math.IsNaN(y) {
		return math.IsNaN(x) && math.IsNaN(y)
	}
	return x == y
}

// equalMessage compares two messages.
func equalMessage(mx, my Message) bool {
	if mx.Descriptor() != my.Descriptor() {
		return false
	}

	nx := 0
	equal := true
	mx.Range(func(fd FieldDescriptor, vx Value) bool {
		nx++
		vy := my.Get(fd)
		equal = my.Has(fd) && equalValue(vx, vy)
		return equal
	})
	if !equal {
		return false
	}
	ny := 0
	my.Range(func(fd FieldDescriptor, vx Value) bool {
		ny++
		return true
	})
	if nx != ny {
		return false
	}

	return equalUnknown(mx.GetUnknown(), my.GetUnknown())
}

// equalList compares two lists.
func equalList(x, y List) bool {
	if x.Len() != y.Len() {
		return false
	}
	for i := x.Len() - 1; i >= 0; i-- {
		if !equalValue(x.Get(i), y.Get(i)) {
			return false
		}
	}
	return true
}

// equalMap compares two maps.
func equalMap(x, y Map) bool {
	if x.Len() != y.Len() {
		return false
	}
	equal := true
	x.Range(func(k MapKey, vx Value) bool {
		vy := y.Get(k)
		equal = y.Has(k) && equalValue(vx, vy)
		return equal
	})
	return equal
}

// equalUnknown compares unknown fields by direct comparison on the raw bytes
// of each individual field number.
func equalUnknown(x, y RawFields) bool {
	if len(x) != len(y) {
		return false
	}
	if bytes.Equal([]byte(x), []byte(y)) {
		return true
	}

	mx := make(map[FieldNumber]RawFields)
	my := make(map[FieldNumber]RawFields)
	for len(x) > 0 {
		fnum, _, n := protowire.ConsumeField(x)
		mx[fnum] = append(mx[fnum], x[:n]...)
		x = x[n:]
	}
	for len(y) > 0 {
		fnum, _, n := protowire.ConsumeField(y)
		my[fnum] = append(my[fnum], y[:n]...)
		y = y[n:]
	}
	return reflect.DeepEqual(mx, my)
}
