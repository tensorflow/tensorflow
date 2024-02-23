// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package filetype provides functionality for wrapping descriptors
// with Go type information.
package filetype

import (
	"reflect"

	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/filedesc"
	pimpl "google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

// Builder constructs type descriptors from a raw file descriptor
// and associated Go types for each enum and message declaration.
//
// # Flattened Ordering
//
// The protobuf type system represents declarations as a tree. Certain nodes in
// the tree require us to either associate it with a concrete Go type or to
// resolve a dependency, which is information that must be provided separately
// since it cannot be derived from the file descriptor alone.
//
// However, representing a tree as Go literals is difficult to simply do in a
// space and time efficient way. Thus, we store them as a flattened list of
// objects where the serialization order from the tree-based form is important.
//
// The "flattened ordering" is defined as a tree traversal of all enum, message,
// extension, and service declarations using the following algorithm:
//
//	def VisitFileDecls(fd):
//		for e in fd.Enums:      yield e
//		for m in fd.Messages:   yield m
//		for x in fd.Extensions: yield x
//		for s in fd.Services:   yield s
//		for m in fd.Messages:   yield from VisitMessageDecls(m)
//
//	def VisitMessageDecls(md):
//		for e in md.Enums:      yield e
//		for m in md.Messages:   yield m
//		for x in md.Extensions: yield x
//		for m in md.Messages:   yield from VisitMessageDecls(m)
//
// The traversal starts at the root file descriptor and yields each direct
// declaration within each node before traversing into sub-declarations
// that children themselves may have.
type Builder struct {
	// File is the underlying file descriptor builder.
	File filedesc.Builder

	// GoTypes is a unique set of the Go types for all declarations and
	// dependencies. Each type is represented as a zero value of the Go type.
	//
	// Declarations are Go types generated for enums and messages directly
	// declared (not publicly imported) in the proto source file.
	// Messages for map entries are accounted for, but represented by nil.
	// Enum declarations in "flattened ordering" come first, followed by
	// message declarations in "flattened ordering".
	//
	// Dependencies are Go types for enums or messages referenced by
	// message fields (excluding weak fields), for parent extended messages of
	// extension fields, for enums or messages referenced by extension fields,
	// and for input and output messages referenced by service methods.
	// Dependencies must come after declarations, but the ordering of
	// dependencies themselves is unspecified.
	GoTypes []interface{}

	// DependencyIndexes is an ordered list of indexes into GoTypes for the
	// dependencies of messages, extensions, or services.
	//
	// There are 5 sub-lists in "flattened ordering" concatenated back-to-back:
	//	0. Message field dependencies: list of the enum or message type
	//	referred to by every message field.
	//	1. Extension field targets: list of the extended parent message of
	//	every extension.
	//	2. Extension field dependencies: list of the enum or message type
	//	referred to by every extension field.
	//	3. Service method inputs: list of the input message type
	//	referred to by every service method.
	//	4. Service method outputs: list of the output message type
	//	referred to by every service method.
	//
	// The offset into DependencyIndexes for the start of each sub-list
	// is appended to the end in reverse order.
	DependencyIndexes []int32

	// EnumInfos is a list of enum infos in "flattened ordering".
	EnumInfos []pimpl.EnumInfo

	// MessageInfos is a list of message infos in "flattened ordering".
	// If provided, the GoType and PBType for each element is populated.
	//
	// Requirement: len(MessageInfos) == len(Build.Messages)
	MessageInfos []pimpl.MessageInfo

	// ExtensionInfos is a list of extension infos in "flattened ordering".
	// Each element is initialized and registered with the protoregistry package.
	//
	// Requirement: len(LegacyExtensions) == len(Build.Extensions)
	ExtensionInfos []pimpl.ExtensionInfo

	// TypeRegistry is the registry to register each type descriptor.
	// If nil, it uses protoregistry.GlobalTypes.
	TypeRegistry interface {
		RegisterMessage(protoreflect.MessageType) error
		RegisterEnum(protoreflect.EnumType) error
		RegisterExtension(protoreflect.ExtensionType) error
	}
}

// Out is the output of the builder.
type Out struct {
	File protoreflect.FileDescriptor
}

func (tb Builder) Build() (out Out) {
	// Replace the resolver with one that resolves dependencies by index,
	// which is faster and more reliable than relying on the global registry.
	if tb.File.FileRegistry == nil {
		tb.File.FileRegistry = protoregistry.GlobalFiles
	}
	tb.File.FileRegistry = &resolverByIndex{
		goTypes:      tb.GoTypes,
		depIdxs:      tb.DependencyIndexes,
		fileRegistry: tb.File.FileRegistry,
	}

	// Initialize registry if unpopulated.
	if tb.TypeRegistry == nil {
		tb.TypeRegistry = protoregistry.GlobalTypes
	}

	fbOut := tb.File.Build()
	out.File = fbOut.File

	// Process enums.
	enumGoTypes := tb.GoTypes[:len(fbOut.Enums)]
	if len(tb.EnumInfos) != len(fbOut.Enums) {
		panic("mismatching enum lengths")
	}
	if len(fbOut.Enums) > 0 {
		for i := range fbOut.Enums {
			tb.EnumInfos[i] = pimpl.EnumInfo{
				GoReflectType: reflect.TypeOf(enumGoTypes[i]),
				Desc:          &fbOut.Enums[i],
			}
			// Register enum types.
			if err := tb.TypeRegistry.RegisterEnum(&tb.EnumInfos[i]); err != nil {
				panic(err)
			}
		}
	}

	// Process messages.
	messageGoTypes := tb.GoTypes[len(fbOut.Enums):][:len(fbOut.Messages)]
	if len(tb.MessageInfos) != len(fbOut.Messages) {
		panic("mismatching message lengths")
	}
	if len(fbOut.Messages) > 0 {
		for i := range fbOut.Messages {
			if messageGoTypes[i] == nil {
				continue // skip map entry
			}

			tb.MessageInfos[i].GoReflectType = reflect.TypeOf(messageGoTypes[i])
			tb.MessageInfos[i].Desc = &fbOut.Messages[i]

			// Register message types.
			if err := tb.TypeRegistry.RegisterMessage(&tb.MessageInfos[i]); err != nil {
				panic(err)
			}
		}

		// As a special-case for descriptor.proto,
		// locally register concrete message type for the options.
		if out.File.Path() == "google/protobuf/descriptor.proto" && out.File.Package() == "google.protobuf" {
			for i := range fbOut.Messages {
				switch fbOut.Messages[i].Name() {
				case "FileOptions":
					descopts.File = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "EnumOptions":
					descopts.Enum = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "EnumValueOptions":
					descopts.EnumValue = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "MessageOptions":
					descopts.Message = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "FieldOptions":
					descopts.Field = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "OneofOptions":
					descopts.Oneof = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "ExtensionRangeOptions":
					descopts.ExtensionRange = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "ServiceOptions":
					descopts.Service = messageGoTypes[i].(protoreflect.ProtoMessage)
				case "MethodOptions":
					descopts.Method = messageGoTypes[i].(protoreflect.ProtoMessage)
				}
			}
		}
	}

	// Process extensions.
	if len(tb.ExtensionInfos) != len(fbOut.Extensions) {
		panic("mismatching extension lengths")
	}
	var depIdx int32
	for i := range fbOut.Extensions {
		// For enum and message kinds, determine the referent Go type so
		// that we can construct their constructors.
		const listExtDeps = 2
		var goType reflect.Type
		switch fbOut.Extensions[i].L1.Kind {
		case protoreflect.EnumKind:
			j := depIdxs.Get(tb.DependencyIndexes, listExtDeps, depIdx)
			goType = reflect.TypeOf(tb.GoTypes[j])
			depIdx++
		case protoreflect.MessageKind, protoreflect.GroupKind:
			j := depIdxs.Get(tb.DependencyIndexes, listExtDeps, depIdx)
			goType = reflect.TypeOf(tb.GoTypes[j])
			depIdx++
		default:
			goType = goTypeForPBKind[fbOut.Extensions[i].L1.Kind]
		}
		if fbOut.Extensions[i].IsList() {
			goType = reflect.SliceOf(goType)
		}

		pimpl.InitExtensionInfo(&tb.ExtensionInfos[i], &fbOut.Extensions[i], goType)

		// Register extension types.
		if err := tb.TypeRegistry.RegisterExtension(&tb.ExtensionInfos[i]); err != nil {
			panic(err)
		}
	}

	return out
}

var goTypeForPBKind = map[protoreflect.Kind]reflect.Type{
	protoreflect.BoolKind:     reflect.TypeOf(bool(false)),
	protoreflect.Int32Kind:    reflect.TypeOf(int32(0)),
	protoreflect.Sint32Kind:   reflect.TypeOf(int32(0)),
	protoreflect.Sfixed32Kind: reflect.TypeOf(int32(0)),
	protoreflect.Int64Kind:    reflect.TypeOf(int64(0)),
	protoreflect.Sint64Kind:   reflect.TypeOf(int64(0)),
	protoreflect.Sfixed64Kind: reflect.TypeOf(int64(0)),
	protoreflect.Uint32Kind:   reflect.TypeOf(uint32(0)),
	protoreflect.Fixed32Kind:  reflect.TypeOf(uint32(0)),
	protoreflect.Uint64Kind:   reflect.TypeOf(uint64(0)),
	protoreflect.Fixed64Kind:  reflect.TypeOf(uint64(0)),
	protoreflect.FloatKind:    reflect.TypeOf(float32(0)),
	protoreflect.DoubleKind:   reflect.TypeOf(float64(0)),
	protoreflect.StringKind:   reflect.TypeOf(string("")),
	protoreflect.BytesKind:    reflect.TypeOf([]byte(nil)),
}

type depIdxs []int32

// Get retrieves the jth element of the ith sub-list.
func (x depIdxs) Get(i, j int32) int32 {
	return x[x[int32(len(x))-i-1]+j]
}

type (
	resolverByIndex struct {
		goTypes []interface{}
		depIdxs depIdxs
		fileRegistry
	}
	fileRegistry interface {
		FindFileByPath(string) (protoreflect.FileDescriptor, error)
		FindDescriptorByName(protoreflect.FullName) (protoreflect.Descriptor, error)
		RegisterFile(protoreflect.FileDescriptor) error
	}
)

func (r *resolverByIndex) FindEnumByIndex(i, j int32, es []filedesc.Enum, ms []filedesc.Message) protoreflect.EnumDescriptor {
	if depIdx := int(r.depIdxs.Get(i, j)); int(depIdx) < len(es)+len(ms) {
		return &es[depIdx]
	} else {
		return pimpl.Export{}.EnumDescriptorOf(r.goTypes[depIdx])
	}
}

func (r *resolverByIndex) FindMessageByIndex(i, j int32, es []filedesc.Enum, ms []filedesc.Message) protoreflect.MessageDescriptor {
	if depIdx := int(r.depIdxs.Get(i, j)); depIdx < len(es)+len(ms) {
		return &ms[depIdx-len(es)]
	} else {
		return pimpl.Export{}.MessageDescriptorOf(r.goTypes[depIdx])
	}
}
