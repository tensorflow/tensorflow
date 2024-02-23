// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package prototest exercises protobuf reflection.
package prototest

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// TODO: Test invalid field descriptors or oneof descriptors.
// TODO: This should test the functionality that can be provided by fast-paths.

// Message tests a message implementation.
type Message struct {
	// Resolver is used to determine the list of extension fields to test with.
	// If nil, this defaults to using protoregistry.GlobalTypes.
	Resolver interface {
		FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error)
		FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error)
		RangeExtensionsByMessage(message protoreflect.FullName, f func(protoreflect.ExtensionType) bool)
	}
}

// Test performs tests on a [protoreflect.MessageType] implementation.
func (test Message) Test(t testing.TB, mt protoreflect.MessageType) {
	testType(t, mt)

	md := mt.Descriptor()
	m1 := mt.New()
	for i := 0; i < md.Fields().Len(); i++ {
		fd := md.Fields().Get(i)
		testField(t, m1, fd)
	}
	if test.Resolver == nil {
		test.Resolver = protoregistry.GlobalTypes
	}
	var extTypes []protoreflect.ExtensionType
	test.Resolver.RangeExtensionsByMessage(md.FullName(), func(e protoreflect.ExtensionType) bool {
		extTypes = append(extTypes, e)
		return true
	})
	for _, xt := range extTypes {
		testField(t, m1, xt.TypeDescriptor())
	}
	for i := 0; i < md.Oneofs().Len(); i++ {
		testOneof(t, m1, md.Oneofs().Get(i))
	}
	testUnknown(t, m1)

	// Test round-trip marshal/unmarshal.
	m2 := mt.New().Interface()
	populateMessage(m2.ProtoReflect(), 1, nil)
	for _, xt := range extTypes {
		m2.ProtoReflect().Set(xt.TypeDescriptor(), newValue(m2.ProtoReflect(), xt.TypeDescriptor(), 1, nil))
	}
	b, err := proto.MarshalOptions{
		AllowPartial: true,
	}.Marshal(m2)
	if err != nil {
		t.Errorf("Marshal() = %v, want nil\n%v", err, prototext.Format(m2))
	}
	m3 := mt.New().Interface()
	if err := (proto.UnmarshalOptions{
		AllowPartial: true,
		Resolver:     test.Resolver,
	}.Unmarshal(b, m3)); err != nil {
		t.Errorf("Unmarshal() = %v, want nil\n%v", err, prototext.Format(m2))
	}
	if !proto.Equal(m2, m3) {
		t.Errorf("round-trip marshal/unmarshal did not preserve message\nOriginal:\n%v\nNew:\n%v", prototext.Format(m2), prototext.Format(m3))
	}
}

func testType(t testing.TB, mt protoreflect.MessageType) {
	m := mt.New().Interface()
	want := reflect.TypeOf(m)
	if got := reflect.TypeOf(m.ProtoReflect().Interface()); got != want {
		t.Errorf("type mismatch: reflect.TypeOf(m) != reflect.TypeOf(m.ProtoReflect().Interface()): %v != %v", got, want)
	}
	if got := reflect.TypeOf(m.ProtoReflect().New().Interface()); got != want {
		t.Errorf("type mismatch: reflect.TypeOf(m) != reflect.TypeOf(m.ProtoReflect().New().Interface()): %v != %v", got, want)
	}
	if got := reflect.TypeOf(m.ProtoReflect().Type().Zero().Interface()); got != want {
		t.Errorf("type mismatch: reflect.TypeOf(m) != reflect.TypeOf(m.ProtoReflect().Type().Zero().Interface()): %v != %v", got, want)
	}
	if mt, ok := mt.(protoreflect.MessageFieldTypes); ok {
		testFieldTypes(t, mt)
	}
}

func testFieldTypes(t testing.TB, mt protoreflect.MessageFieldTypes) {
	descName := func(d protoreflect.Descriptor) protoreflect.FullName {
		if d == nil {
			return "<nil>"
		}
		return d.FullName()
	}
	typeName := func(mt protoreflect.MessageType) protoreflect.FullName {
		if mt == nil {
			return "<nil>"
		}
		return mt.Descriptor().FullName()
	}
	adjustExpr := func(idx int, expr string) string {
		expr = strings.Replace(expr, "fd.", "md.Fields().Get(i).", -1)
		expr = strings.Replace(expr, "(fd)", "(md.Fields().Get(i))", -1)
		expr = strings.Replace(expr, "mti.", "mt.Message(i).", -1)
		expr = strings.Replace(expr, "(i)", fmt.Sprintf("(%d)", idx), -1)
		return expr
	}
	checkEnumDesc := func(idx int, gotExpr, wantExpr string, got, want protoreflect.EnumDescriptor) {
		if got != want {
			t.Errorf("descriptor mismatch: %v != %v: %v != %v", adjustExpr(idx, gotExpr), adjustExpr(idx, wantExpr), descName(got), descName(want))
		}
	}
	checkMessageDesc := func(idx int, gotExpr, wantExpr string, got, want protoreflect.MessageDescriptor) {
		if got != want {
			t.Errorf("descriptor mismatch: %v != %v: %v != %v", adjustExpr(idx, gotExpr), adjustExpr(idx, wantExpr), descName(got), descName(want))
		}
	}
	checkMessageType := func(idx int, gotExpr, wantExpr string, got, want protoreflect.MessageType) {
		if got != want {
			t.Errorf("type mismatch: %v != %v: %v != %v", adjustExpr(idx, gotExpr), adjustExpr(idx, wantExpr), typeName(got), typeName(want))
		}
	}

	fds := mt.Descriptor().Fields()
	m := mt.New()
	for i := 0; i < fds.Len(); i++ {
		fd := fds.Get(i)
		switch {
		case fd.IsList():
			if fd.Enum() != nil {
				checkEnumDesc(i,
					"mt.Enum(i).Descriptor()", "fd.Enum()",
					mt.Enum(i).Descriptor(), fd.Enum())
			}
			if fd.Message() != nil {
				checkMessageDesc(i,
					"mt.Message(i).Descriptor()", "fd.Message()",
					mt.Message(i).Descriptor(), fd.Message())
				checkMessageType(i,
					"mt.Message(i)", "m.NewField(fd).List().NewElement().Message().Type()",
					mt.Message(i), m.NewField(fd).List().NewElement().Message().Type())
			}
		case fd.IsMap():
			mti := mt.Message(i)
			if m := mti.New(); m != nil {
				checkMessageDesc(i,
					"m.Descriptor()", "fd.Message()",
					m.Descriptor(), fd.Message())
			}
			if m := mti.Zero(); m != nil {
				checkMessageDesc(i,
					"m.Descriptor()", "fd.Message()",
					m.Descriptor(), fd.Message())
			}
			checkMessageDesc(i,
				"mti.Descriptor()", "fd.Message()",
				mti.Descriptor(), fd.Message())
			if mti := mti.(protoreflect.MessageFieldTypes); mti != nil {
				if fd.MapValue().Enum() != nil {
					checkEnumDesc(i,
						"mti.Enum(fd.MapValue().Index()).Descriptor()", "fd.MapValue().Enum()",
						mti.Enum(fd.MapValue().Index()).Descriptor(), fd.MapValue().Enum())
				}
				if fd.MapValue().Message() != nil {
					checkMessageDesc(i,
						"mti.Message(fd.MapValue().Index()).Descriptor()", "fd.MapValue().Message()",
						mti.Message(fd.MapValue().Index()).Descriptor(), fd.MapValue().Message())
					checkMessageType(i,
						"mti.Message(fd.MapValue().Index())", "m.NewField(fd).Map().NewValue().Message().Type()",
						mti.Message(fd.MapValue().Index()), m.NewField(fd).Map().NewValue().Message().Type())
				}
			}
		default:
			if fd.Enum() != nil {
				checkEnumDesc(i,
					"mt.Enum(i).Descriptor()", "fd.Enum()",
					mt.Enum(i).Descriptor(), fd.Enum())
			}
			if fd.Message() != nil {
				checkMessageDesc(i,
					"mt.Message(i).Descriptor()", "fd.Message()",
					mt.Message(i).Descriptor(), fd.Message())
				checkMessageType(i,
					"mt.Message(i)", "m.NewField(fd).Message().Type()",
					mt.Message(i), m.NewField(fd).Message().Type())
			}
		}
	}
}

// testField exercises set/get/has/clear of a field.
func testField(t testing.TB, m protoreflect.Message, fd protoreflect.FieldDescriptor) {
	name := fd.FullName()
	num := fd.Number()

	switch {
	case fd.IsList():
		testFieldList(t, m, fd)
	case fd.IsMap():
		testFieldMap(t, m, fd)
	case fd.Message() != nil:
	default:
		if got, want := m.NewField(fd), fd.Default(); !valueEqual(got, want) {
			t.Errorf("Message.NewField(%v) = %v, want default value %v", name, formatValue(got), formatValue(want))
		}
		if fd.Kind() == protoreflect.FloatKind || fd.Kind() == protoreflect.DoubleKind {
			testFieldFloat(t, m, fd)
		}
	}

	// Set to a non-zero value, the zero value, different non-zero values.
	for _, n := range []seed{1, 0, minVal, maxVal} {
		v := newValue(m, fd, n, nil)
		m.Set(fd, v)
		wantHas := true
		if n == 0 {
			if !fd.HasPresence() {
				wantHas = false
			}
			if fd.IsExtension() {
				wantHas = true
			}
			if fd.Cardinality() == protoreflect.Repeated {
				wantHas = false
			}
			if fd.ContainingOneof() != nil {
				wantHas = true
			}
		}
		if !fd.HasPresence() && fd.Cardinality() != protoreflect.Repeated && fd.ContainingOneof() == nil && fd.Kind() == protoreflect.EnumKind && v.Enum() == 0 {
			wantHas = false
		}
		if got, want := m.Has(fd), wantHas; got != want {
			t.Errorf("after setting %q to %v:\nMessage.Has(%v) = %v, want %v", name, formatValue(v), num, got, want)
		}
		if got, want := m.Get(fd), v; !valueEqual(got, want) {
			t.Errorf("after setting %q:\nMessage.Get(%v) = %v, want %v", name, num, formatValue(got), formatValue(want))
		}
		found := false
		m.Range(func(d protoreflect.FieldDescriptor, got protoreflect.Value) bool {
			if fd != d {
				return true
			}
			found = true
			if want := v; !valueEqual(got, want) {
				t.Errorf("after setting %q:\nMessage.Range got value %v, want %v", name, formatValue(got), formatValue(want))
			}
			return true
		})
		if got, want := wantHas, found; got != want {
			t.Errorf("after setting %q:\nMessageRange saw field: %v, want %v", name, got, want)
		}
	}

	m.Clear(fd)
	if got, want := m.Has(fd), false; got != want {
		t.Errorf("after clearing %q:\nMessage.Has(%v) = %v, want %v", name, num, got, want)
	}
	switch {
	case fd.IsList():
		if got := m.Get(fd); got.List().Len() != 0 {
			t.Errorf("after clearing %q:\nMessage.Get(%v) = %v, want empty list", name, num, formatValue(got))
		}
	case fd.IsMap():
		if got := m.Get(fd); got.Map().Len() != 0 {
			t.Errorf("after clearing %q:\nMessage.Get(%v) = %v, want empty map", name, num, formatValue(got))
		}
	case fd.Message() == nil:
		if got, want := m.Get(fd), fd.Default(); !valueEqual(got, want) {
			t.Errorf("after clearing %q:\nMessage.Get(%v) = %v, want default %v", name, num, formatValue(got), formatValue(want))
		}
	}

	// Set to the default value.
	switch {
	case fd.IsList() || fd.IsMap():
		m.Set(fd, m.Mutable(fd))
		if got, want := m.Has(fd), (fd.IsExtension() && fd.Cardinality() != protoreflect.Repeated) || fd.ContainingOneof() != nil; got != want {
			t.Errorf("after setting %q to default:\nMessage.Has(%v) = %v, want %v", name, num, got, want)
		}
	case fd.Message() == nil:
		m.Set(fd, m.Get(fd))
		if got, want := m.Get(fd), fd.Default(); !valueEqual(got, want) {
			t.Errorf("after setting %q to default:\nMessage.Get(%v) = %v, want default %v", name, num, formatValue(got), formatValue(want))
		}
	}
	m.Clear(fd)

	// Set to the wrong type.
	v := protoreflect.ValueOfString("")
	if fd.Kind() == protoreflect.StringKind {
		v = protoreflect.ValueOfInt32(0)
	}
	if !panics(func() {
		m.Set(fd, v)
	}) {
		t.Errorf("setting %v to %T succeeds, want panic", name, v.Interface())
	}
}

// testFieldMap tests set/get/has/clear of entries in a map field.
func testFieldMap(t testing.TB, m protoreflect.Message, fd protoreflect.FieldDescriptor) {
	name := fd.FullName()
	num := fd.Number()

	// New values.
	m.Clear(fd) // start with an empty map
	mapv := m.Get(fd).Map()
	if mapv.IsValid() {
		t.Errorf("after clearing field: message.Get(%v).IsValid() = true, want false", name)
	}
	if got, want := mapv.NewValue(), newMapValue(fd, mapv, 0, nil); !valueEqual(got, want) {
		t.Errorf("message.Get(%v).NewValue() = %v, want %v", name, formatValue(got), formatValue(want))
	}
	if !panics(func() {
		m.Set(fd, protoreflect.ValueOfMap(mapv))
	}) {
		t.Errorf("message.Set(%v, <invalid>) does not panic", name)
	}
	if !panics(func() {
		mapv.Set(newMapKey(fd, 0), newMapValue(fd, mapv, 0, nil))
	}) {
		t.Errorf("message.Get(%v).Set(...) of invalid map does not panic", name)
	}
	mapv = m.Mutable(fd).Map() // mutable map
	if !mapv.IsValid() {
		t.Errorf("message.Mutable(%v).IsValid() = false, want true", name)
	}
	if got, want := mapv.NewValue(), newMapValue(fd, mapv, 0, nil); !valueEqual(got, want) {
		t.Errorf("message.Mutable(%v).NewValue() = %v, want %v", name, formatValue(got), formatValue(want))
	}

	// Add values.
	want := make(testMap)
	for i, n := range []seed{1, 0, minVal, maxVal} {
		if got, want := m.Has(fd), i > 0; got != want {
			t.Errorf("after inserting %d elements to %q:\nMessage.Has(%v) = %v, want %v", i, name, num, got, want)
		}

		k := newMapKey(fd, n)
		v := newMapValue(fd, mapv, n, nil)
		mapv.Set(k, v)
		want.Set(k, v)
		if got, want := m.Get(fd), protoreflect.ValueOfMap(want); !valueEqual(got, want) {
			t.Errorf("after inserting %d elements to %q:\nMessage.Get(%v) = %v, want %v", i, name, num, formatValue(got), formatValue(want))
		}
	}

	// Set values.
	want.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		nv := newMapValue(fd, mapv, 10, nil)
		mapv.Set(k, nv)
		want.Set(k, nv)
		if got, want := m.Get(fd), protoreflect.ValueOfMap(want); !valueEqual(got, want) {
			t.Errorf("after setting element %v of %q:\nMessage.Get(%v) = %v, want %v", formatValue(k.Value()), name, num, formatValue(got), formatValue(want))
		}
		return true
	})

	// Clear values.
	want.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		mapv.Clear(k)
		want.Clear(k)
		if got, want := m.Has(fd), want.Len() > 0; got != want {
			t.Errorf("after clearing elements of %q:\nMessage.Has(%v) = %v, want %v", name, num, got, want)
		}
		if got, want := m.Get(fd), protoreflect.ValueOfMap(want); !valueEqual(got, want) {
			t.Errorf("after clearing elements of %q:\nMessage.Get(%v) = %v, want %v", name, num, formatValue(got), formatValue(want))
		}
		return true
	})
	if mapv := m.Get(fd).Map(); mapv.IsValid() {
		t.Errorf("after clearing all elements: message.Get(%v).IsValid() = true, want false %v", name, formatValue(protoreflect.ValueOfMap(mapv)))
	}

	// Non-existent map keys.
	missingKey := newMapKey(fd, 1)
	if got, want := mapv.Has(missingKey), false; got != want {
		t.Errorf("non-existent map key in %q: Map.Has(%v) = %v, want %v", name, formatValue(missingKey.Value()), got, want)
	}
	if got, want := mapv.Get(missingKey).IsValid(), false; got != want {
		t.Errorf("non-existent map key in %q: Map.Get(%v).IsValid() = %v, want %v", name, formatValue(missingKey.Value()), got, want)
	}
	mapv.Clear(missingKey) // noop

	// Mutable.
	if fd.MapValue().Message() == nil {
		if !panics(func() {
			mapv.Mutable(newMapKey(fd, 1))
		}) {
			t.Errorf("Mutable on %q succeeds, want panic", name)
		}
	} else {
		k := newMapKey(fd, 1)
		v := mapv.Mutable(k)
		if got, want := mapv.Len(), 1; got != want {
			t.Errorf("after Mutable on %q, Map.Len() = %v, want %v", name, got, want)
		}
		populateMessage(v.Message(), 1, nil)
		if !valueEqual(mapv.Get(k), v) {
			t.Errorf("after Mutable on %q, changing new mutable value does not change map entry", name)
		}
		mapv.Clear(k)
	}
}

type testMap map[interface{}]protoreflect.Value

func (m testMap) Get(k protoreflect.MapKey) protoreflect.Value     { return m[k.Interface()] }
func (m testMap) Set(k protoreflect.MapKey, v protoreflect.Value)  { m[k.Interface()] = v }
func (m testMap) Has(k protoreflect.MapKey) bool                   { return m.Get(k).IsValid() }
func (m testMap) Clear(k protoreflect.MapKey)                      { delete(m, k.Interface()) }
func (m testMap) Mutable(k protoreflect.MapKey) protoreflect.Value { panic("unimplemented") }
func (m testMap) Len() int                                         { return len(m) }
func (m testMap) NewValue() protoreflect.Value                     { panic("unimplemented") }
func (m testMap) Range(f func(protoreflect.MapKey, protoreflect.Value) bool) {
	for k, v := range m {
		if !f(protoreflect.ValueOf(k).MapKey(), v) {
			return
		}
	}
}
func (m testMap) IsValid() bool { return true }

// testFieldList exercises set/get/append/truncate of values in a list.
func testFieldList(t testing.TB, m protoreflect.Message, fd protoreflect.FieldDescriptor) {
	name := fd.FullName()
	num := fd.Number()

	m.Clear(fd) // start with an empty list
	list := m.Get(fd).List()
	if list.IsValid() {
		t.Errorf("message.Get(%v).IsValid() = true, want false", name)
	}
	if !panics(func() {
		m.Set(fd, protoreflect.ValueOfList(list))
	}) {
		t.Errorf("message.Set(%v, <invalid>) does not panic", name)
	}
	if !panics(func() {
		list.Append(newListElement(fd, list, 0, nil))
	}) {
		t.Errorf("message.Get(%v).Append(...) of invalid list does not panic", name)
	}
	if got, want := list.NewElement(), newListElement(fd, list, 0, nil); !valueEqual(got, want) {
		t.Errorf("message.Get(%v).NewElement() = %v, want %v", name, formatValue(got), formatValue(want))
	}
	list = m.Mutable(fd).List() // mutable list
	if !list.IsValid() {
		t.Errorf("message.Get(%v).IsValid() = false, want true", name)
	}
	if got, want := list.NewElement(), newListElement(fd, list, 0, nil); !valueEqual(got, want) {
		t.Errorf("message.Mutable(%v).NewElement() = %v, want %v", name, formatValue(got), formatValue(want))
	}

	// Append values.
	var want protoreflect.List = &testList{}
	for i, n := range []seed{1, 0, minVal, maxVal} {
		if got, want := m.Has(fd), i > 0; got != want {
			t.Errorf("after appending %d elements to %q:\nMessage.Has(%v) = %v, want %v", i, name, num, got, want)
		}
		v := newListElement(fd, list, n, nil)
		want.Append(v)
		list.Append(v)

		if got, want := m.Get(fd), protoreflect.ValueOfList(want); !valueEqual(got, want) {
			t.Errorf("after appending %d elements to %q:\nMessage.Get(%v) = %v, want %v", i+1, name, num, formatValue(got), formatValue(want))
		}
	}

	// Set values.
	for i := 0; i < want.Len(); i++ {
		v := newListElement(fd, list, seed(i+10), nil)
		want.Set(i, v)
		list.Set(i, v)
		if got, want := m.Get(fd), protoreflect.ValueOfList(want); !valueEqual(got, want) {
			t.Errorf("after setting element %d of %q:\nMessage.Get(%v) = %v, want %v", i, name, num, formatValue(got), formatValue(want))
		}
	}

	// Truncate.
	for want.Len() > 0 {
		n := want.Len() - 1
		want.Truncate(n)
		list.Truncate(n)
		if got, want := m.Has(fd), want.Len() > 0; got != want {
			t.Errorf("after truncating %q to %d:\nMessage.Has(%v) = %v, want %v", name, n, num, got, want)
		}
		if got, want := m.Get(fd), protoreflect.ValueOfList(want); !valueEqual(got, want) {
			t.Errorf("after truncating %q to %d:\nMessage.Get(%v) = %v, want %v", name, n, num, formatValue(got), formatValue(want))
		}
	}

	// AppendMutable.
	if fd.Message() == nil {
		if !panics(func() {
			list.AppendMutable()
		}) {
			t.Errorf("AppendMutable on %q succeeds, want panic", name)
		}
	} else {
		v := list.AppendMutable()
		if got, want := list.Len(), 1; got != want {
			t.Errorf("after AppendMutable on %q, list.Len() = %v, want %v", name, got, want)
		}
		populateMessage(v.Message(), 1, nil)
		if !valueEqual(list.Get(0), v) {
			t.Errorf("after AppendMutable on %q, changing new mutable value does not change list item 0", name)
		}
		want.Truncate(0)
	}
}

type testList struct {
	a []protoreflect.Value
}

func (l *testList) Append(v protoreflect.Value)       { l.a = append(l.a, v) }
func (l *testList) AppendMutable() protoreflect.Value { panic("unimplemented") }
func (l *testList) Get(n int) protoreflect.Value      { return l.a[n] }
func (l *testList) Len() int                          { return len(l.a) }
func (l *testList) Set(n int, v protoreflect.Value)   { l.a[n] = v }
func (l *testList) Truncate(n int)                    { l.a = l.a[:n] }
func (l *testList) NewElement() protoreflect.Value    { panic("unimplemented") }
func (l *testList) IsValid() bool                     { return true }

// testFieldFloat exercises some interesting floating-point scalar field values.
func testFieldFloat(t testing.TB, m protoreflect.Message, fd protoreflect.FieldDescriptor) {
	name := fd.FullName()
	num := fd.Number()

	for _, v := range []float64{math.Inf(-1), math.Inf(1), math.NaN(), math.Copysign(0, -1)} {
		var val protoreflect.Value
		if fd.Kind() == protoreflect.FloatKind {
			val = protoreflect.ValueOfFloat32(float32(v))
		} else {
			val = protoreflect.ValueOfFloat64(float64(v))
		}
		m.Set(fd, val)
		// Note that Has is true for -0.
		if got, want := m.Has(fd), true; got != want {
			t.Errorf("after setting %v to %v: Message.Has(%v) = %v, want %v", name, v, num, got, want)
		}
		if got, want := m.Get(fd), val; !valueEqual(got, want) {
			t.Errorf("after setting %v: Message.Get(%v) = %v, want %v", name, num, formatValue(got), formatValue(want))
		}
	}
}

// testOneof tests the behavior of fields in a oneof.
func testOneof(t testing.TB, m protoreflect.Message, od protoreflect.OneofDescriptor) {
	for _, mutable := range []bool{false, true} {
		for i := 0; i < od.Fields().Len(); i++ {
			fda := od.Fields().Get(i)
			if mutable {
				// Set fields by requesting a mutable reference.
				if !fda.IsMap() && !fda.IsList() && fda.Message() == nil {
					continue
				}
				_ = m.Mutable(fda)
			} else {
				// Set fields explicitly.
				m.Set(fda, newValue(m, fda, 1, nil))
			}
			if got, want := m.WhichOneof(od), fda; got != want {
				t.Errorf("after setting oneof field %q:\nWhichOneof(%q) = %v, want %v", fda.FullName(), fda.Name(), got, want)
			}
			for j := 0; j < od.Fields().Len(); j++ {
				fdb := od.Fields().Get(j)
				if got, want := m.Has(fdb), i == j; got != want {
					t.Errorf("after setting oneof field %q:\nGet(%q) = %v, want %v", fda.FullName(), fdb.FullName(), got, want)
				}
			}
		}
	}
}

// testUnknown tests the behavior of unknown fields.
func testUnknown(t testing.TB, m protoreflect.Message) {
	var b []byte
	b = protowire.AppendTag(b, 1000, protowire.VarintType)
	b = protowire.AppendVarint(b, 1001)
	m.SetUnknown(protoreflect.RawFields(b))
	if got, want := []byte(m.GetUnknown()), b; !bytes.Equal(got, want) {
		t.Errorf("after setting unknown fields:\nGetUnknown() = %v, want %v", got, want)
	}
}

func formatValue(v protoreflect.Value) string {
	switch v := v.Interface().(type) {
	case protoreflect.List:
		var buf bytes.Buffer
		buf.WriteString("list[")
		for i := 0; i < v.Len(); i++ {
			if i > 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(formatValue(v.Get(i)))
		}
		buf.WriteString("]")
		return buf.String()
	case protoreflect.Map:
		var buf bytes.Buffer
		buf.WriteString("map[")
		var keys []protoreflect.MapKey
		v.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
			keys = append(keys, k)
			return true
		})
		sort.Slice(keys, func(i, j int) bool {
			return keys[i].String() < keys[j].String()
		})
		for i, k := range keys {
			if i > 0 {
				buf.WriteString(" ")
			}
			buf.WriteString(formatValue(k.Value()))
			buf.WriteString(":")
			buf.WriteString(formatValue(v.Get(k)))
		}
		buf.WriteString("]")
		return buf.String()
	case protoreflect.Message:
		b, err := prototext.Marshal(v.Interface())
		if err != nil {
			return fmt.Sprintf("<%v>", err)
		}
		return fmt.Sprintf("%v{%s}", v.Descriptor().FullName(), b)
	case string:
		return fmt.Sprintf("%q", v)
	default:
		return fmt.Sprint(v)
	}
}

func valueEqual(a, b protoreflect.Value) bool {
	ai, bi := a.Interface(), b.Interface()
	switch ai.(type) {
	case protoreflect.Message:
		return proto.Equal(
			a.Message().Interface(),
			b.Message().Interface(),
		)
	case protoreflect.List:
		lista, listb := a.List(), b.List()
		if lista.Len() != listb.Len() {
			return false
		}
		for i := 0; i < lista.Len(); i++ {
			if !valueEqual(lista.Get(i), listb.Get(i)) {
				return false
			}
		}
		return true
	case protoreflect.Map:
		mapa, mapb := a.Map(), b.Map()
		if mapa.Len() != mapb.Len() {
			return false
		}
		equal := true
		mapa.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
			if !valueEqual(v, mapb.Get(k)) {
				equal = false
				return false
			}
			return true
		})
		return equal
	case []byte:
		return bytes.Equal(a.Bytes(), b.Bytes())
	case float32:
		// NaNs are equal, but must be the same NaN.
		return math.Float32bits(ai.(float32)) == math.Float32bits(bi.(float32))
	case float64:
		// NaNs are equal, but must be the same NaN.
		return math.Float64bits(ai.(float64)) == math.Float64bits(bi.(float64))
	default:
		return ai == bi
	}
}

// A seed is used to vary the content of a value.
//
// A seed of 0 is the zero value. Messages do not have a zero-value; a 0-seeded messages
// is unpopulated.
//
// A seed of minVal or maxVal is the least or greatest value of the value type.
type seed int

const (
	minVal seed = -1
	maxVal seed = -2
)

// newSeed creates new seed values from a base, for example to create seeds for the
// elements in a list. If the input seed is minVal or maxVal, so is the output.
func newSeed(n seed, adjust ...int) seed {
	switch n {
	case minVal, maxVal:
		return n
	}
	for _, a := range adjust {
		n = 10*n + seed(a)
	}
	return n
}

// newValue returns a new value assignable to a field.
//
// The stack parameter is used to avoid infinite recursion when populating circular
// data structures.
func newValue(m protoreflect.Message, fd protoreflect.FieldDescriptor, n seed, stack []protoreflect.MessageDescriptor) protoreflect.Value {
	switch {
	case fd.IsList():
		if n == 0 {
			return m.New().Mutable(fd)
		}
		list := m.NewField(fd).List()
		list.Append(newListElement(fd, list, 0, stack))
		list.Append(newListElement(fd, list, minVal, stack))
		list.Append(newListElement(fd, list, maxVal, stack))
		list.Append(newListElement(fd, list, n, stack))
		return protoreflect.ValueOfList(list)
	case fd.IsMap():
		if n == 0 {
			return m.New().Mutable(fd)
		}
		mapv := m.NewField(fd).Map()
		mapv.Set(newMapKey(fd, 0), newMapValue(fd, mapv, 0, stack))
		mapv.Set(newMapKey(fd, minVal), newMapValue(fd, mapv, minVal, stack))
		mapv.Set(newMapKey(fd, maxVal), newMapValue(fd, mapv, maxVal, stack))
		mapv.Set(newMapKey(fd, n), newMapValue(fd, mapv, newSeed(n, 0), stack))
		return protoreflect.ValueOfMap(mapv)
	case fd.Message() != nil:
		return populateMessage(m.NewField(fd).Message(), n, stack)
	default:
		return newScalarValue(fd, n)
	}
}

func newListElement(fd protoreflect.FieldDescriptor, list protoreflect.List, n seed, stack []protoreflect.MessageDescriptor) protoreflect.Value {
	if fd.Message() == nil {
		return newScalarValue(fd, n)
	}
	return populateMessage(list.NewElement().Message(), n, stack)
}

func newMapKey(fd protoreflect.FieldDescriptor, n seed) protoreflect.MapKey {
	kd := fd.MapKey()
	return newScalarValue(kd, n).MapKey()
}

func newMapValue(fd protoreflect.FieldDescriptor, mapv protoreflect.Map, n seed, stack []protoreflect.MessageDescriptor) protoreflect.Value {
	vd := fd.MapValue()
	if vd.Message() == nil {
		return newScalarValue(vd, n)
	}
	return populateMessage(mapv.NewValue().Message(), n, stack)
}

func newScalarValue(fd protoreflect.FieldDescriptor, n seed) protoreflect.Value {
	switch fd.Kind() {
	case protoreflect.BoolKind:
		return protoreflect.ValueOfBool(n != 0)
	case protoreflect.EnumKind:
		vals := fd.Enum().Values()
		var i int
		switch n {
		case minVal:
			i = 0
		case maxVal:
			i = vals.Len() - 1
		default:
			i = int(n) % vals.Len()
		}
		return protoreflect.ValueOfEnum(vals.Get(i).Number())
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
		switch n {
		case minVal:
			return protoreflect.ValueOfInt32(math.MinInt32)
		case maxVal:
			return protoreflect.ValueOfInt32(math.MaxInt32)
		default:
			return protoreflect.ValueOfInt32(int32(n))
		}
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
		switch n {
		case minVal:
			// Only use 0 for the zero value.
			return protoreflect.ValueOfUint32(1)
		case maxVal:
			return protoreflect.ValueOfUint32(math.MaxInt32)
		default:
			return protoreflect.ValueOfUint32(uint32(n))
		}
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
		switch n {
		case minVal:
			return protoreflect.ValueOfInt64(math.MinInt64)
		case maxVal:
			return protoreflect.ValueOfInt64(math.MaxInt64)
		default:
			return protoreflect.ValueOfInt64(int64(n))
		}
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
		switch n {
		case minVal:
			// Only use 0 for the zero value.
			return protoreflect.ValueOfUint64(1)
		case maxVal:
			return protoreflect.ValueOfUint64(math.MaxInt64)
		default:
			return protoreflect.ValueOfUint64(uint64(n))
		}
	case protoreflect.FloatKind:
		switch n {
		case minVal:
			return protoreflect.ValueOfFloat32(math.SmallestNonzeroFloat32)
		case maxVal:
			return protoreflect.ValueOfFloat32(math.MaxFloat32)
		default:
			return protoreflect.ValueOfFloat32(1.5 * float32(n))
		}
	case protoreflect.DoubleKind:
		switch n {
		case minVal:
			return protoreflect.ValueOfFloat64(math.SmallestNonzeroFloat64)
		case maxVal:
			return protoreflect.ValueOfFloat64(math.MaxFloat64)
		default:
			return protoreflect.ValueOfFloat64(1.5 * float64(n))
		}
	case protoreflect.StringKind:
		if n == 0 {
			return protoreflect.ValueOfString("")
		}
		return protoreflect.ValueOfString(fmt.Sprintf("%d", n))
	case protoreflect.BytesKind:
		if n == 0 {
			return protoreflect.ValueOfBytes(nil)
		}
		return protoreflect.ValueOfBytes([]byte{byte(n >> 24), byte(n >> 16), byte(n >> 8), byte(n)})
	}
	panic("unhandled kind")
}

func populateMessage(m protoreflect.Message, n seed, stack []protoreflect.MessageDescriptor) protoreflect.Value {
	if n == 0 {
		return protoreflect.ValueOfMessage(m)
	}
	md := m.Descriptor()
	for _, x := range stack {
		if md == x {
			return protoreflect.ValueOfMessage(m)
		}
	}
	stack = append(stack, md)
	for i := 0; i < md.Fields().Len(); i++ {
		fd := md.Fields().Get(i)
		if fd.IsWeak() {
			continue
		}
		m.Set(fd, newValue(m, fd, newSeed(n, i), stack))
	}
	return protoreflect.ValueOfMessage(m)
}

func panics(f func()) (didPanic bool) {
	defer func() {
		if err := recover(); err != nil {
			didPanic = true
		}
	}()
	f()
	return false
}
