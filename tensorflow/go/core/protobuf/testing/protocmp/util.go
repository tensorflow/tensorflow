// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protocmp

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strings"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

var (
	enumReflectType    = reflect.TypeOf(Enum{})
	messageReflectType = reflect.TypeOf(Message{})
)

// FilterEnum filters opt to only be applicable on a standalone [Enum],
// singular fields of enums, list fields of enums, or map fields of enum values,
// where the enum is the same type as the specified enum.
//
// The Go type of the last path step may be an:
//   - [Enum] for singular fields, elements of a repeated field,
//     values of a map field, or standalone [Enum] values
//   - [][Enum] for list fields
//   - map[K][Enum] for map fields
//   - interface{} for a [Message] map entry value
//
// This must be used in conjunction with [Transform].
func FilterEnum(enum protoreflect.Enum, opt cmp.Option) cmp.Option {
	return FilterDescriptor(enum.Descriptor(), opt)
}

// FilterMessage filters opt to only be applicable on a standalone [Message] values,
// singular fields of messages, list fields of messages, or map fields of
// message values, where the message is the same type as the specified message.
//
// The Go type of the last path step may be an:
//   - [Message] for singular fields, elements of a repeated field,
//     values of a map field, or standalone [Message] values
//   - [][Message] for list fields
//   - map[K][Message] for map fields
//   - interface{} for a [Message] map entry value
//
// This must be used in conjunction with [Transform].
func FilterMessage(message proto.Message, opt cmp.Option) cmp.Option {
	return FilterDescriptor(message.ProtoReflect().Descriptor(), opt)
}

// FilterField filters opt to only be applicable on the specified field
// in the message. It panics if a field of the given name does not exist.
//
// The Go type of the last path step may be an:
//   - T for singular fields
//   - []T for list fields
//   - map[K]T for map fields
//   - interface{} for a [Message] map entry value
//
// This must be used in conjunction with [Transform].
func FilterField(message proto.Message, name protoreflect.Name, opt cmp.Option) cmp.Option {
	md := message.ProtoReflect().Descriptor()
	return FilterDescriptor(mustFindFieldDescriptor(md, name), opt)
}

// FilterOneof filters opt to only be applicable on all fields within the
// specified oneof in the message. It panics if a oneof of the given name
// does not exist.
//
// The Go type of the last path step may be an:
//   - T for singular fields
//   - []T for list fields
//   - map[K]T for map fields
//   - interface{} for a [Message] map entry value
//
// This must be used in conjunction with [Transform].
func FilterOneof(message proto.Message, name protoreflect.Name, opt cmp.Option) cmp.Option {
	md := message.ProtoReflect().Descriptor()
	return FilterDescriptor(mustFindOneofDescriptor(md, name), opt)
}

// FilterDescriptor ignores the specified descriptor.
//
// The following descriptor types may be specified:
//   - [protoreflect.EnumDescriptor]
//   - [protoreflect.MessageDescriptor]
//   - [protoreflect.FieldDescriptor]
//   - [protoreflect.OneofDescriptor]
//
// For the behavior of each, see the corresponding filter function.
// Since this filter accepts a [protoreflect.FieldDescriptor], it can be used
// to also filter for extension fields as a [protoreflect.ExtensionDescriptor]
// is just an alias to [protoreflect.FieldDescriptor].
//
// This must be used in conjunction with [Transform].
func FilterDescriptor(desc protoreflect.Descriptor, opt cmp.Option) cmp.Option {
	f := newNameFilters(desc)
	return cmp.FilterPath(f.Filter, opt)
}

// IgnoreEnums ignores all enums of the specified types.
// It is equivalent to FilterEnum(enum, cmp.Ignore()) for each enum.
//
// This must be used in conjunction with [Transform].
func IgnoreEnums(enums ...protoreflect.Enum) cmp.Option {
	var ds []protoreflect.Descriptor
	for _, e := range enums {
		ds = append(ds, e.Descriptor())
	}
	return IgnoreDescriptors(ds...)
}

// IgnoreMessages ignores all messages of the specified types.
// It is equivalent to [FilterMessage](message, [cmp.Ignore]()) for each message.
//
// This must be used in conjunction with [Transform].
func IgnoreMessages(messages ...proto.Message) cmp.Option {
	var ds []protoreflect.Descriptor
	for _, m := range messages {
		ds = append(ds, m.ProtoReflect().Descriptor())
	}
	return IgnoreDescriptors(ds...)
}

// IgnoreFields ignores the specified fields in the specified message.
// It is equivalent to [FilterField](message, name, [cmp.Ignore]()) for each field
// in the message.
//
// This must be used in conjunction with [Transform].
func IgnoreFields(message proto.Message, names ...protoreflect.Name) cmp.Option {
	var ds []protoreflect.Descriptor
	md := message.ProtoReflect().Descriptor()
	for _, s := range names {
		ds = append(ds, mustFindFieldDescriptor(md, s))
	}
	return IgnoreDescriptors(ds...)
}

// IgnoreOneofs ignores fields of the specified oneofs in the specified message.
// It is equivalent to FilterOneof(message, name, cmp.Ignore()) for each oneof
// in the message.
//
// This must be used in conjunction with [Transform].
func IgnoreOneofs(message proto.Message, names ...protoreflect.Name) cmp.Option {
	var ds []protoreflect.Descriptor
	md := message.ProtoReflect().Descriptor()
	for _, s := range names {
		ds = append(ds, mustFindOneofDescriptor(md, s))
	}
	return IgnoreDescriptors(ds...)
}

// IgnoreDescriptors ignores the specified set of descriptors.
// It is equivalent to [FilterDescriptor](desc, [cmp.Ignore]()) for each descriptor.
//
// This must be used in conjunction with [Transform].
func IgnoreDescriptors(descs ...protoreflect.Descriptor) cmp.Option {
	return cmp.FilterPath(newNameFilters(descs...).Filter, cmp.Ignore())
}

func mustFindFieldDescriptor(md protoreflect.MessageDescriptor, s protoreflect.Name) protoreflect.FieldDescriptor {
	d := findDescriptor(md, s)
	if fd, ok := d.(protoreflect.FieldDescriptor); ok && fd.TextName() == string(s) {
		return fd
	}

	var suggestion string
	switch d := d.(type) {
	case protoreflect.FieldDescriptor:
		suggestion = fmt.Sprintf("; consider specifying field %q instead", d.TextName())
	case protoreflect.OneofDescriptor:
		suggestion = fmt.Sprintf("; consider specifying oneof %q with IgnoreOneofs instead", d.Name())
	}
	panic(fmt.Sprintf("message %q has no field %q%s", md.FullName(), s, suggestion))
}

func mustFindOneofDescriptor(md protoreflect.MessageDescriptor, s protoreflect.Name) protoreflect.OneofDescriptor {
	d := findDescriptor(md, s)
	if od, ok := d.(protoreflect.OneofDescriptor); ok && d.Name() == s {
		return od
	}

	var suggestion string
	switch d := d.(type) {
	case protoreflect.OneofDescriptor:
		suggestion = fmt.Sprintf("; consider specifying oneof %q instead", d.Name())
	case protoreflect.FieldDescriptor:
		suggestion = fmt.Sprintf("; consider specifying field %q with IgnoreFields instead", d.TextName())
	}
	panic(fmt.Sprintf("message %q has no oneof %q%s", md.FullName(), s, suggestion))
}

func findDescriptor(md protoreflect.MessageDescriptor, s protoreflect.Name) protoreflect.Descriptor {
	// Exact match.
	if fd := md.Fields().ByTextName(string(s)); fd != nil {
		return fd
	}
	if od := md.Oneofs().ByName(s); od != nil && !od.IsSynthetic() {
		return od
	}

	// Best-effort match.
	//
	// It's a common user mistake to use the CamelCased field name as it appears
	// in the generated Go struct. Instead of complaining that it doesn't exist,
	// suggest the real protobuf name that the user may have desired.
	normalize := func(s protoreflect.Name) string {
		return strings.Replace(strings.ToLower(string(s)), "_", "", -1)
	}
	for i := 0; i < md.Fields().Len(); i++ {
		if fd := md.Fields().Get(i); normalize(fd.Name()) == normalize(s) {
			return fd
		}
	}
	for i := 0; i < md.Oneofs().Len(); i++ {
		if od := md.Oneofs().Get(i); normalize(od.Name()) == normalize(s) {
			return od
		}
	}
	return nil
}

type nameFilters struct {
	names map[protoreflect.FullName]bool
}

func newNameFilters(descs ...protoreflect.Descriptor) *nameFilters {
	f := &nameFilters{names: make(map[protoreflect.FullName]bool)}
	for _, d := range descs {
		switch d := d.(type) {
		case protoreflect.EnumDescriptor:
			f.names[d.FullName()] = true
		case protoreflect.MessageDescriptor:
			f.names[d.FullName()] = true
		case protoreflect.FieldDescriptor:
			f.names[d.FullName()] = true
		case protoreflect.OneofDescriptor:
			for i := 0; i < d.Fields().Len(); i++ {
				f.names[d.Fields().Get(i).FullName()] = true
			}
		default:
			panic("invalid descriptor type")
		}
	}
	return f
}

func (f *nameFilters) Filter(p cmp.Path) bool {
	vx, vy := p.Last().Values()
	return (f.filterValue(vx) && f.filterValue(vy)) || f.filterFields(p)
}

func (f *nameFilters) filterFields(p cmp.Path) bool {
	// Trim off trailing type-assertions so that the filter can match on the
	// concrete value held within an interface value.
	if _, ok := p.Last().(cmp.TypeAssertion); ok {
		p = p[:len(p)-1]
	}

	// Filter for Message maps.
	mi, ok := p.Index(-1).(cmp.MapIndex)
	if !ok {
		return false
	}
	ps := p.Index(-2)
	if ps.Type() != messageReflectType {
		return false
	}

	// Check field name.
	vx, vy := ps.Values()
	mx := vx.Interface().(Message)
	my := vy.Interface().(Message)
	k := mi.Key().String()
	if f.filterFieldName(mx, k) && f.filterFieldName(my, k) {
		return true
	}

	// Check field value.
	vx, vy = mi.Values()
	if f.filterFieldValue(vx) && f.filterFieldValue(vy) {
		return true
	}

	return false
}

func (f *nameFilters) filterFieldName(m Message, k string) bool {
	if _, ok := m[k]; !ok {
		return true // treat missing fields as already filtered
	}
	var fd protoreflect.FieldDescriptor
	switch mm := m[messageTypeKey].(messageMeta); {
	case protoreflect.Name(k).IsValid():
		fd = mm.md.Fields().ByTextName(k)
	default:
		fd = mm.xds[k]
	}
	if fd != nil {
		return f.names[fd.FullName()]
	}
	return false
}

func (f *nameFilters) filterFieldValue(v reflect.Value) bool {
	if !v.IsValid() {
		return true // implies missing slice element or map entry
	}
	v = v.Elem() // map entries are always populated values
	switch t := v.Type(); {
	case t == enumReflectType || t == messageReflectType:
		// Check for singular message or enum field.
		return f.filterValue(v)
	case t.Kind() == reflect.Slice && (t.Elem() == enumReflectType || t.Elem() == messageReflectType):
		// Check for list field of enum or message type.
		return f.filterValue(v.Index(0))
	case t.Kind() == reflect.Map && (t.Elem() == enumReflectType || t.Elem() == messageReflectType):
		// Check for map field of enum or message type.
		return f.filterValue(v.MapIndex(v.MapKeys()[0]))
	}
	return false
}

func (f *nameFilters) filterValue(v reflect.Value) bool {
	if !v.IsValid() {
		return true // implies missing slice element or map entry
	}
	if !v.CanInterface() {
		return false // implies unexported struct field
	}
	switch v := v.Interface().(type) {
	case Enum:
		return v.Descriptor() != nil && f.names[v.Descriptor().FullName()]
	case Message:
		return v.Descriptor() != nil && f.names[v.Descriptor().FullName()]
	}
	return false
}

// IgnoreDefaultScalars ignores singular scalars that are unpopulated or
// explicitly set to the default value.
// This option does not effect elements in a list or entries in a map.
//
// This must be used in conjunction with [Transform].
func IgnoreDefaultScalars() cmp.Option {
	return cmp.FilterPath(func(p cmp.Path) bool {
		// Filter for Message maps.
		mi, ok := p.Index(-1).(cmp.MapIndex)
		if !ok {
			return false
		}
		ps := p.Index(-2)
		if ps.Type() != messageReflectType {
			return false
		}

		// Check whether both fields are default or unpopulated scalars.
		vx, vy := ps.Values()
		mx := vx.Interface().(Message)
		my := vy.Interface().(Message)
		k := mi.Key().String()
		return isDefaultScalar(mx, k) && isDefaultScalar(my, k)
	}, cmp.Ignore())
}

func isDefaultScalar(m Message, k string) bool {
	if _, ok := m[k]; !ok {
		return true
	}

	var fd protoreflect.FieldDescriptor
	switch mm := m[messageTypeKey].(messageMeta); {
	case protoreflect.Name(k).IsValid():
		fd = mm.md.Fields().ByTextName(k)
	default:
		fd = mm.xds[k]
	}
	if fd == nil || !fd.Default().IsValid() {
		return false
	}
	switch fd.Kind() {
	case protoreflect.BytesKind:
		v, ok := m[k].([]byte)
		return ok && bytes.Equal(fd.Default().Bytes(), v)
	case protoreflect.FloatKind:
		v, ok := m[k].(float32)
		return ok && equalFloat64(fd.Default().Float(), float64(v))
	case protoreflect.DoubleKind:
		v, ok := m[k].(float64)
		return ok && equalFloat64(fd.Default().Float(), float64(v))
	case protoreflect.EnumKind:
		v, ok := m[k].(Enum)
		return ok && fd.Default().Enum() == v.Number()
	default:
		return reflect.DeepEqual(fd.Default().Interface(), m[k])
	}
}

func equalFloat64(x, y float64) bool {
	return x == y || (math.IsNaN(x) && math.IsNaN(y))
}

// IgnoreEmptyMessages ignores messages that are empty or unpopulated.
// It applies to standalone [Message] values, singular message fields,
// list fields of messages, and map fields of message values.
//
// This must be used in conjunction with [Transform].
func IgnoreEmptyMessages() cmp.Option {
	return cmp.FilterPath(func(p cmp.Path) bool {
		vx, vy := p.Last().Values()
		return (isEmptyMessage(vx) && isEmptyMessage(vy)) || isEmptyMessageFields(p)
	}, cmp.Ignore())
}

func isEmptyMessageFields(p cmp.Path) bool {
	// Filter for Message maps.
	mi, ok := p.Index(-1).(cmp.MapIndex)
	if !ok {
		return false
	}
	ps := p.Index(-2)
	if ps.Type() != messageReflectType {
		return false
	}

	// Check field value.
	vx, vy := mi.Values()
	if isEmptyMessageFieldValue(vx) && isEmptyMessageFieldValue(vy) {
		return true
	}

	return false
}

func isEmptyMessageFieldValue(v reflect.Value) bool {
	if !v.IsValid() {
		return true // implies missing slice element or map entry
	}
	v = v.Elem() // map entries are always populated values
	switch t := v.Type(); {
	case t == messageReflectType:
		// Check singular field for empty message.
		if !isEmptyMessage(v) {
			return false
		}
	case t.Kind() == reflect.Slice && t.Elem() == messageReflectType:
		// Check list field for all empty message elements.
		for i := 0; i < v.Len(); i++ {
			if !isEmptyMessage(v.Index(i)) {
				return false
			}
		}
	case t.Kind() == reflect.Map && t.Elem() == messageReflectType:
		// Check map field for all empty message values.
		for _, k := range v.MapKeys() {
			if !isEmptyMessage(v.MapIndex(k)) {
				return false
			}
		}
	default:
		return false
	}
	return true
}

func isEmptyMessage(v reflect.Value) bool {
	if !v.IsValid() {
		return true // implies missing slice element or map entry
	}
	if !v.CanInterface() {
		return false // implies unexported struct field
	}
	if m, ok := v.Interface().(Message); ok {
		for k := range m {
			if k != messageTypeKey && k != messageInvalidKey {
				return false
			}
		}
		return true
	}
	return false
}

// IgnoreUnknown ignores unknown fields in all messages.
//
// This must be used in conjunction with [Transform].
func IgnoreUnknown() cmp.Option {
	return cmp.FilterPath(func(p cmp.Path) bool {
		// Filter for Message maps.
		mi, ok := p.Index(-1).(cmp.MapIndex)
		if !ok {
			return false
		}
		ps := p.Index(-2)
		if ps.Type() != messageReflectType {
			return false
		}

		// Filter for unknown fields (which always have a numeric map key).
		return strings.Trim(mi.Key().String(), "0123456789") == ""
	}, cmp.Ignore())
}

// SortRepeated sorts repeated fields of the specified element type.
// The less function must be of the form "func(T, T) bool" where T is the
// Go element type for the repeated field kind.
//
// The element type T can be one of the following:
//   - Go type for a protobuf scalar kind except for an enum
//     (i.e., bool, int32, int64, uint32, uint64, float32, float64, string, and []byte)
//   - E where E is a concrete enum type that implements [protoreflect.Enum]
//   - M where M is a concrete message type that implement [proto.Message]
//
// This option only applies to repeated fields within a protobuf message.
// It does not operate on higher-order Go types that seem like a repeated field.
// For example, a []T outside the context of a protobuf message will not be
// handled by this option. To sort Go slices that are not repeated fields,
// consider using [github.com/google/go-cmp/cmp/cmpopts.SortSlices] instead.
//
// This must be used in conjunction with [Transform].
func SortRepeated(lessFunc interface{}) cmp.Option {
	t, ok := checkTTBFunc(lessFunc)
	if !ok {
		panic(fmt.Sprintf("invalid less function: %T", lessFunc))
	}

	var opt cmp.Option
	var sliceType reflect.Type
	switch vf := reflect.ValueOf(lessFunc); {
	case t.Implements(enumV2Type):
		et := reflect.Zero(t).Interface().(protoreflect.Enum).Type()
		lessFunc = func(x, y Enum) bool {
			vx := reflect.ValueOf(et.New(x.Number()))
			vy := reflect.ValueOf(et.New(y.Number()))
			return vf.Call([]reflect.Value{vx, vy})[0].Bool()
		}
		opt = FilterDescriptor(et.Descriptor(), cmpopts.SortSlices(lessFunc))
		sliceType = reflect.SliceOf(enumReflectType)
	case t.Implements(messageV2Type):
		mt := reflect.Zero(t).Interface().(protoreflect.ProtoMessage).ProtoReflect().Type()
		lessFunc = func(x, y Message) bool {
			mx := mt.New().Interface()
			my := mt.New().Interface()
			proto.Merge(mx, x)
			proto.Merge(my, y)
			vx := reflect.ValueOf(mx)
			vy := reflect.ValueOf(my)
			return vf.Call([]reflect.Value{vx, vy})[0].Bool()
		}
		opt = FilterDescriptor(mt.Descriptor(), cmpopts.SortSlices(lessFunc))
		sliceType = reflect.SliceOf(messageReflectType)
	default:
		switch t {
		case reflect.TypeOf(bool(false)):
		case reflect.TypeOf(int32(0)):
		case reflect.TypeOf(int64(0)):
		case reflect.TypeOf(uint32(0)):
		case reflect.TypeOf(uint64(0)):
		case reflect.TypeOf(float32(0)):
		case reflect.TypeOf(float64(0)):
		case reflect.TypeOf(string("")):
		case reflect.TypeOf([]byte(nil)):
		default:
			panic(fmt.Sprintf("invalid element type: %v", t))
		}
		opt = cmpopts.SortSlices(lessFunc)
		sliceType = reflect.SliceOf(t)
	}

	return cmp.FilterPath(func(p cmp.Path) bool {
		// Filter to only apply to repeated fields within a message.
		if t := p.Index(-1).Type(); t == nil || t != sliceType {
			return false
		}
		if t := p.Index(-2).Type(); t == nil || t.Kind() != reflect.Interface {
			return false
		}
		if t := p.Index(-3).Type(); t == nil || t != messageReflectType {
			return false
		}
		return true
	}, opt)
}

func checkTTBFunc(lessFunc interface{}) (reflect.Type, bool) {
	switch t := reflect.TypeOf(lessFunc); {
	case t == nil:
		return nil, false
	case t.NumIn() != 2 || t.In(0) != t.In(1) || t.IsVariadic():
		return nil, false
	case t.NumOut() != 1 || t.Out(0) != reflect.TypeOf(false):
		return nil, false
	default:
		return t.In(0), true
	}
}

// SortRepeatedFields sorts the specified repeated fields.
// Sorting a repeated field is useful for treating the list as a multiset
// (i.e., a set where each value can appear multiple times).
// It panics if the field does not exist or is not a repeated field.
//
// The sort ordering is as follows:
//   - Booleans are sorted where false is sorted before true.
//   - Integers are sorted in ascending order.
//   - Floating-point numbers are sorted in ascending order according to
//     the total ordering defined by IEEE-754 (section 5.10).
//   - Strings and bytes are sorted lexicographically in ascending order.
//   - [Enum] values are sorted in ascending order based on its numeric value.
//   - [Message] values are sorted according to some arbitrary ordering
//     which is undefined and may change in future implementations.
//
// The ordering chosen for repeated messages is unlikely to be aesthetically
// preferred by humans. Consider using a custom sort function:
//
//	FilterField(m, "foo_field", SortRepeated(func(x, y *foopb.MyMessage) bool {
//	    ... // user-provided definition for less
//	}))
//
// This must be used in conjunction with [Transform].
func SortRepeatedFields(message proto.Message, names ...protoreflect.Name) cmp.Option {
	var opts cmp.Options
	md := message.ProtoReflect().Descriptor()
	for _, name := range names {
		fd := mustFindFieldDescriptor(md, name)
		if !fd.IsList() {
			panic(fmt.Sprintf("message field %q is not repeated", fd.FullName()))
		}

		var lessFunc interface{}
		switch fd.Kind() {
		case protoreflect.BoolKind:
			lessFunc = func(x, y bool) bool { return !x && y }
		case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind:
			lessFunc = func(x, y int32) bool { return x < y }
		case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind:
			lessFunc = func(x, y int64) bool { return x < y }
		case protoreflect.Uint32Kind, protoreflect.Fixed32Kind:
			lessFunc = func(x, y uint32) bool { return x < y }
		case protoreflect.Uint64Kind, protoreflect.Fixed64Kind:
			lessFunc = func(x, y uint64) bool { return x < y }
		case protoreflect.FloatKind:
			lessFunc = lessF32
		case protoreflect.DoubleKind:
			lessFunc = lessF64
		case protoreflect.StringKind:
			lessFunc = func(x, y string) bool { return x < y }
		case protoreflect.BytesKind:
			lessFunc = func(x, y []byte) bool { return bytes.Compare(x, y) < 0 }
		case protoreflect.EnumKind:
			lessFunc = func(x, y Enum) bool { return x.Number() < y.Number() }
		case protoreflect.MessageKind, protoreflect.GroupKind:
			lessFunc = func(x, y Message) bool { return x.String() < y.String() }
		default:
			panic(fmt.Sprintf("invalid kind: %v", fd.Kind()))
		}
		opts = append(opts, FilterDescriptor(fd, cmpopts.SortSlices(lessFunc)))
	}
	return opts
}

func lessF32(x, y float32) bool {
	// Bit-wise implementation of IEEE-754, section 5.10.
	xi := int32(math.Float32bits(x))
	yi := int32(math.Float32bits(y))
	xi ^= int32(uint32(xi>>31) >> 1)
	yi ^= int32(uint32(yi>>31) >> 1)
	return xi < yi
}
func lessF64(x, y float64) bool {
	// Bit-wise implementation of IEEE-754, section 5.10.
	xi := int64(math.Float64bits(x))
	yi := int64(math.Float64bits(y))
	xi ^= int64(uint64(xi>>63) >> 1)
	yi ^= int64(uint64(yi>>63) >> 1)
	return xi < yi
}
