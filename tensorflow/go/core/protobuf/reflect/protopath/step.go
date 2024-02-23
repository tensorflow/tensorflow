// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protopath

import (
	"fmt"
	"strconv"
	"strings"

	"google.golang.org/protobuf/internal/encoding/text"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// StepKind identifies the kind of step operation.
// Each kind of step corresponds with some protobuf reflection operation.
type StepKind int

const (
	invalidStep StepKind = iota
	// RootStep identifies a step as the Root step operation.
	RootStep
	// FieldAccessStep identifies a step as the FieldAccess step operation.
	FieldAccessStep
	// UnknownAccessStep identifies a step as the UnknownAccess step operation.
	UnknownAccessStep
	// ListIndexStep identifies a step as the ListIndex step operation.
	ListIndexStep
	// MapIndexStep identifies a step as the MapIndex step operation.
	MapIndexStep
	// AnyExpandStep identifies a step as the AnyExpand step operation.
	AnyExpandStep
)

func (k StepKind) String() string {
	switch k {
	case invalidStep:
		return "<invalid>"
	case RootStep:
		return "Root"
	case FieldAccessStep:
		return "FieldAccess"
	case UnknownAccessStep:
		return "UnknownAccess"
	case ListIndexStep:
		return "ListIndex"
	case MapIndexStep:
		return "MapIndex"
	case AnyExpandStep:
		return "AnyExpand"
	default:
		return fmt.Sprintf("<unknown:%d>", k)
	}
}

// Step is a union where only one step operation may be specified at a time.
// The different kinds of steps are specified by the constants defined for
// the StepKind type.
type Step struct {
	kind StepKind
	desc protoreflect.Descriptor
	key  protoreflect.Value
}

// Root indicates the root message that a path is relative to.
// It should always (and only ever) be the first step in a path.
func Root(md protoreflect.MessageDescriptor) Step {
	if md == nil {
		panic("nil message descriptor")
	}
	return Step{kind: RootStep, desc: md}
}

// FieldAccess describes access of a field within a message.
// Extension field accesses are also represented using a FieldAccess and
// must be provided with a protoreflect.FieldDescriptor
//
// Within the context of Values,
// the type of the previous step value is always a message, and
// the type of the current step value is determined by the field descriptor.
func FieldAccess(fd protoreflect.FieldDescriptor) Step {
	if fd == nil {
		panic("nil field descriptor")
	} else if _, ok := fd.(protoreflect.ExtensionTypeDescriptor); !ok && fd.IsExtension() {
		panic(fmt.Sprintf("extension field %q must implement protoreflect.ExtensionTypeDescriptor", fd.FullName()))
	}
	return Step{kind: FieldAccessStep, desc: fd}
}

// UnknownAccess describes access to the unknown fields within a message.
//
// Within the context of Values,
// the type of the previous step value is always a message, and
// the type of the current step value is always a bytes type.
func UnknownAccess() Step {
	return Step{kind: UnknownAccessStep}
}

// ListIndex describes index of an element within a list.
//
// Within the context of Values,
// the type of the previous, previous step value is always a message,
// the type of the previous step value is always a list, and
// the type of the current step value is determined by the field descriptor.
func ListIndex(i int) Step {
	if i < 0 {
		panic(fmt.Sprintf("invalid list index: %v", i))
	}
	return Step{kind: ListIndexStep, key: protoreflect.ValueOfInt64(int64(i))}
}

// MapIndex describes index of an entry within a map.
// The key type is determined by field descriptor that the map belongs to.
//
// Within the context of Values,
// the type of the previous previous step value is always a message,
// the type of the previous step value is always a map, and
// the type of the current step value is determined by the field descriptor.
func MapIndex(k protoreflect.MapKey) Step {
	if !k.IsValid() {
		panic("invalid map index")
	}
	return Step{kind: MapIndexStep, key: k.Value()}
}

// AnyExpand describes expansion of a google.protobuf.Any message into
// a structured representation of the underlying message.
//
// Within the context of Values,
// the type of the previous step value is always a google.protobuf.Any message, and
// the type of the current step value is always a message.
func AnyExpand(md protoreflect.MessageDescriptor) Step {
	if md == nil {
		panic("nil message descriptor")
	}
	return Step{kind: AnyExpandStep, desc: md}
}

// MessageDescriptor returns the message descriptor for Root or AnyExpand steps,
// otherwise it returns nil.
func (s Step) MessageDescriptor() protoreflect.MessageDescriptor {
	switch s.kind {
	case RootStep, AnyExpandStep:
		return s.desc.(protoreflect.MessageDescriptor)
	default:
		return nil
	}
}

// FieldDescriptor returns the field descriptor for FieldAccess steps,
// otherwise it returns nil.
func (s Step) FieldDescriptor() protoreflect.FieldDescriptor {
	switch s.kind {
	case FieldAccessStep:
		return s.desc.(protoreflect.FieldDescriptor)
	default:
		return nil
	}
}

// ListIndex returns the list index for ListIndex steps,
// otherwise it returns 0.
func (s Step) ListIndex() int {
	switch s.kind {
	case ListIndexStep:
		return int(s.key.Int())
	default:
		return 0
	}
}

// MapIndex returns the map key for MapIndex steps,
// otherwise it returns an invalid map key.
func (s Step) MapIndex() protoreflect.MapKey {
	switch s.kind {
	case MapIndexStep:
		return s.key.MapKey()
	default:
		return protoreflect.MapKey{}
	}
}

// Kind reports which kind of step this is.
func (s Step) Kind() StepKind {
	return s.kind
}

func (s Step) String() string {
	return string(s.appendString(nil))
}

func (s Step) appendString(b []byte) []byte {
	switch s.kind {
	case RootStep:
		b = append(b, '(')
		b = append(b, s.desc.FullName()...)
		b = append(b, ')')
	case FieldAccessStep:
		b = append(b, '.')
		if fd := s.desc.(protoreflect.FieldDescriptor); fd.IsExtension() {
			b = append(b, '(')
			b = append(b, strings.Trim(fd.TextName(), "[]")...)
			b = append(b, ')')
		} else {
			b = append(b, fd.TextName()...)
		}
	case UnknownAccessStep:
		b = append(b, '.')
		b = append(b, '?')
	case ListIndexStep:
		b = append(b, '[')
		b = strconv.AppendInt(b, s.key.Int(), 10)
		b = append(b, ']')
	case MapIndexStep:
		b = append(b, '[')
		switch k := s.key.Interface().(type) {
		case bool:
			b = strconv.AppendBool(b, bool(k)) // e.g., "true" or "false"
		case int32:
			b = strconv.AppendInt(b, int64(k), 10) // e.g., "-32"
		case int64:
			b = strconv.AppendInt(b, int64(k), 10) // e.g., "-64"
		case uint32:
			b = strconv.AppendUint(b, uint64(k), 10) // e.g., "32"
		case uint64:
			b = strconv.AppendUint(b, uint64(k), 10) // e.g., "64"
		case string:
			b = text.AppendString(b, k) // e.g., `"hello, world"`
		}
		b = append(b, ']')
	case AnyExpandStep:
		b = append(b, '.')
		b = append(b, '(')
		b = append(b, s.desc.FullName()...)
		b = append(b, ')')
	default:
		b = append(b, "<invalid>"...)
	}
	return b
}
