// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"strings"
	"unicode"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

func validateEnumDeclarations(es []filedesc.Enum, eds []*descriptorpb.EnumDescriptorProto) error {
	for i, ed := range eds {
		e := &es[i]
		if err := e.L2.ReservedNames.CheckValid(); err != nil {
			return errors.New("enum %q reserved names has %v", e.FullName(), err)
		}
		if err := e.L2.ReservedRanges.CheckValid(); err != nil {
			return errors.New("enum %q reserved ranges has %v", e.FullName(), err)
		}
		if len(ed.GetValue()) == 0 {
			return errors.New("enum %q must contain at least one value declaration", e.FullName())
		}
		allowAlias := ed.GetOptions().GetAllowAlias()
		foundAlias := false
		for i := 0; i < e.Values().Len(); i++ {
			v1 := e.Values().Get(i)
			if v2 := e.Values().ByNumber(v1.Number()); v1 != v2 {
				foundAlias = true
				if !allowAlias {
					return errors.New("enum %q has conflicting non-aliased values on number %d: %q with %q", e.FullName(), v1.Number(), v1.Name(), v2.Name())
				}
			}
		}
		if allowAlias && !foundAlias {
			return errors.New("enum %q allows aliases, but none were found", e.FullName())
		}
		if e.Syntax() == protoreflect.Proto3 {
			if v := e.Values().Get(0); v.Number() != 0 {
				return errors.New("enum %q using proto3 semantics must have zero number for the first value", v.FullName())
			}
			// Verify that value names in proto3 do not conflict if the
			// case-insensitive prefix is removed.
			// See protoc v3.8.0: src/google/protobuf/descriptor.cc:4991-5055
			names := map[string]protoreflect.EnumValueDescriptor{}
			prefix := strings.Replace(strings.ToLower(string(e.Name())), "_", "", -1)
			for i := 0; i < e.Values().Len(); i++ {
				v1 := e.Values().Get(i)
				s := strs.EnumValueName(strs.TrimEnumPrefix(string(v1.Name()), prefix))
				if v2, ok := names[s]; ok && v1.Number() != v2.Number() {
					return errors.New("enum %q using proto3 semantics has conflict: %q with %q", e.FullName(), v1.Name(), v2.Name())
				}
				names[s] = v1
			}
		}

		for j, vd := range ed.GetValue() {
			v := &e.L2.Values.List[j]
			if vd.Number == nil {
				return errors.New("enum value %q must have a specified number", v.FullName())
			}
			if e.L2.ReservedNames.Has(v.Name()) {
				return errors.New("enum value %q must not use reserved name", v.FullName())
			}
			if e.L2.ReservedRanges.Has(v.Number()) {
				return errors.New("enum value %q must not use reserved number %d", v.FullName(), v.Number())
			}
		}
	}
	return nil
}

func validateMessageDeclarations(ms []filedesc.Message, mds []*descriptorpb.DescriptorProto) error {
	for i, md := range mds {
		m := &ms[i]

		// Handle the message descriptor itself.
		isMessageSet := md.GetOptions().GetMessageSetWireFormat()
		if err := m.L2.ReservedNames.CheckValid(); err != nil {
			return errors.New("message %q reserved names has %v", m.FullName(), err)
		}
		if err := m.L2.ReservedRanges.CheckValid(isMessageSet); err != nil {
			return errors.New("message %q reserved ranges has %v", m.FullName(), err)
		}
		if err := m.L2.ExtensionRanges.CheckValid(isMessageSet); err != nil {
			return errors.New("message %q extension ranges has %v", m.FullName(), err)
		}
		if err := (*filedesc.FieldRanges).CheckOverlap(&m.L2.ReservedRanges, &m.L2.ExtensionRanges); err != nil {
			return errors.New("message %q reserved and extension ranges has %v", m.FullName(), err)
		}
		for i := 0; i < m.Fields().Len(); i++ {
			f1 := m.Fields().Get(i)
			if f2 := m.Fields().ByNumber(f1.Number()); f1 != f2 {
				return errors.New("message %q has conflicting fields: %q with %q", m.FullName(), f1.Name(), f2.Name())
			}
		}
		if isMessageSet && !flags.ProtoLegacy {
			return errors.New("message %q is a MessageSet, which is a legacy proto1 feature that is no longer supported", m.FullName())
		}
		if isMessageSet && (m.Syntax() != protoreflect.Proto2 || m.Fields().Len() > 0 || m.ExtensionRanges().Len() == 0) {
			return errors.New("message %q is an invalid proto1 MessageSet", m.FullName())
		}
		if m.Syntax() == protoreflect.Proto3 {
			if m.ExtensionRanges().Len() > 0 {
				return errors.New("message %q using proto3 semantics cannot have extension ranges", m.FullName())
			}
			// Verify that field names in proto3 do not conflict if lowercased
			// with all underscores removed.
			// See protoc v3.8.0: src/google/protobuf/descriptor.cc:5830-5847
			names := map[string]protoreflect.FieldDescriptor{}
			for i := 0; i < m.Fields().Len(); i++ {
				f1 := m.Fields().Get(i)
				s := strings.Replace(strings.ToLower(string(f1.Name())), "_", "", -1)
				if f2, ok := names[s]; ok {
					return errors.New("message %q using proto3 semantics has conflict: %q with %q", m.FullName(), f1.Name(), f2.Name())
				}
				names[s] = f1
			}
		}

		for j, fd := range md.GetField() {
			f := &m.L2.Fields.List[j]
			if m.L2.ReservedNames.Has(f.Name()) {
				return errors.New("message field %q must not use reserved name", f.FullName())
			}
			if !f.Number().IsValid() {
				return errors.New("message field %q has an invalid number: %d", f.FullName(), f.Number())
			}
			if !f.Cardinality().IsValid() {
				return errors.New("message field %q has an invalid cardinality: %d", f.FullName(), f.Cardinality())
			}
			if m.L2.ReservedRanges.Has(f.Number()) {
				return errors.New("message field %q must not use reserved number %d", f.FullName(), f.Number())
			}
			if m.L2.ExtensionRanges.Has(f.Number()) {
				return errors.New("message field %q with number %d in extension range", f.FullName(), f.Number())
			}
			if fd.Extendee != nil {
				return errors.New("message field %q may not have extendee: %q", f.FullName(), fd.GetExtendee())
			}
			if f.L1.IsProto3Optional {
				if f.Syntax() != protoreflect.Proto3 {
					return errors.New("message field %q under proto3 optional semantics must be specified in the proto3 syntax", f.FullName())
				}
				if f.Cardinality() != protoreflect.Optional {
					return errors.New("message field %q under proto3 optional semantics must have optional cardinality", f.FullName())
				}
				if f.ContainingOneof() != nil && f.ContainingOneof().Fields().Len() != 1 {
					return errors.New("message field %q under proto3 optional semantics must be within a single element oneof", f.FullName())
				}
			}
			if f.IsWeak() && !flags.ProtoLegacy {
				return errors.New("message field %q is a weak field, which is a legacy proto1 feature that is no longer supported", f.FullName())
			}
			if f.IsWeak() && (f.Syntax() != protoreflect.Proto2 || !isOptionalMessage(f) || f.ContainingOneof() != nil) {
				return errors.New("message field %q may only be weak for an optional message", f.FullName())
			}
			if f.IsPacked() && !isPackable(f) {
				return errors.New("message field %q is not packable", f.FullName())
			}
			if err := checkValidGroup(f); err != nil {
				return errors.New("message field %q is an invalid group: %v", f.FullName(), err)
			}
			if err := checkValidMap(f); err != nil {
				return errors.New("message field %q is an invalid map: %v", f.FullName(), err)
			}
			if f.Syntax() == protoreflect.Proto3 {
				if f.Cardinality() == protoreflect.Required {
					return errors.New("message field %q using proto3 semantics cannot be required", f.FullName())
				}
				if f.Enum() != nil && !f.Enum().IsPlaceholder() && f.Enum().Syntax() != protoreflect.Proto3 {
					return errors.New("message field %q using proto3 semantics may only depend on a proto3 enum", f.FullName())
				}
			}
		}
		seenSynthetic := false // synthetic oneofs for proto3 optional must come after real oneofs
		for j := range md.GetOneofDecl() {
			o := &m.L2.Oneofs.List[j]
			if o.Fields().Len() == 0 {
				return errors.New("message oneof %q must contain at least one field declaration", o.FullName())
			}
			if n := o.Fields().Len(); n-1 != (o.Fields().Get(n-1).Index() - o.Fields().Get(0).Index()) {
				return errors.New("message oneof %q must have consecutively declared fields", o.FullName())
			}

			if o.IsSynthetic() {
				seenSynthetic = true
				continue
			}
			if !o.IsSynthetic() && seenSynthetic {
				return errors.New("message oneof %q must be declared before synthetic oneofs", o.FullName())
			}

			for i := 0; i < o.Fields().Len(); i++ {
				f := o.Fields().Get(i)
				if f.Cardinality() != protoreflect.Optional {
					return errors.New("message field %q belongs in a oneof and must be optional", f.FullName())
				}
				if f.IsWeak() {
					return errors.New("message field %q belongs in a oneof and must not be a weak reference", f.FullName())
				}
			}
		}

		if err := validateEnumDeclarations(m.L1.Enums.List, md.GetEnumType()); err != nil {
			return err
		}
		if err := validateMessageDeclarations(m.L1.Messages.List, md.GetNestedType()); err != nil {
			return err
		}
		if err := validateExtensionDeclarations(m.L1.Extensions.List, md.GetExtension()); err != nil {
			return err
		}
	}
	return nil
}

func validateExtensionDeclarations(xs []filedesc.Extension, xds []*descriptorpb.FieldDescriptorProto) error {
	for i, xd := range xds {
		x := &xs[i]
		// NOTE: Avoid using the IsValid method since extensions to MessageSet
		// may have a field number higher than normal. This check only verifies
		// that the number is not negative or reserved. We check again later
		// if we know that the extendee is definitely not a MessageSet.
		if n := x.Number(); n < 0 || (protowire.FirstReservedNumber <= n && n <= protowire.LastReservedNumber) {
			return errors.New("extension field %q has an invalid number: %d", x.FullName(), x.Number())
		}
		if !x.Cardinality().IsValid() || x.Cardinality() == protoreflect.Required {
			return errors.New("extension field %q has an invalid cardinality: %d", x.FullName(), x.Cardinality())
		}
		if xd.JsonName != nil {
			// A bug in older versions of protoc would always populate the
			// "json_name" option for extensions when it is meaningless.
			// When it did so, it would always use the camel-cased field name.
			if xd.GetJsonName() != strs.JSONCamelCase(string(x.Name())) {
				return errors.New("extension field %q may not have an explicitly set JSON name: %q", x.FullName(), xd.GetJsonName())
			}
		}
		if xd.OneofIndex != nil {
			return errors.New("extension field %q may not be part of a oneof", x.FullName())
		}
		if md := x.ContainingMessage(); !md.IsPlaceholder() {
			if !md.ExtensionRanges().Has(x.Number()) {
				return errors.New("extension field %q extends %q with non-extension field number: %d", x.FullName(), md.FullName(), x.Number())
			}
			isMessageSet := md.Options().(*descriptorpb.MessageOptions).GetMessageSetWireFormat()
			if isMessageSet && !isOptionalMessage(x) {
				return errors.New("extension field %q extends MessageSet and must be an optional message", x.FullName())
			}
			if !isMessageSet && !x.Number().IsValid() {
				return errors.New("extension field %q has an invalid number: %d", x.FullName(), x.Number())
			}
		}
		if xd.GetOptions().GetWeak() {
			return errors.New("extension field %q cannot be a weak reference", x.FullName())
		}
		if x.IsPacked() && !isPackable(x) {
			return errors.New("extension field %q is not packable", x.FullName())
		}
		if err := checkValidGroup(x); err != nil {
			return errors.New("extension field %q is an invalid group: %v", x.FullName(), err)
		}
		if md := x.Message(); md != nil && md.IsMapEntry() {
			return errors.New("extension field %q cannot be a map entry", x.FullName())
		}
		if x.Syntax() == protoreflect.Proto3 {
			switch x.ContainingMessage().FullName() {
			case (*descriptorpb.FileOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.EnumOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.EnumValueOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.MessageOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.FieldOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.OneofOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.ExtensionRangeOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.ServiceOptions)(nil).ProtoReflect().Descriptor().FullName():
			case (*descriptorpb.MethodOptions)(nil).ProtoReflect().Descriptor().FullName():
			default:
				return errors.New("extension field %q cannot be declared in proto3 unless extended descriptor options", x.FullName())
			}
		}
	}
	return nil
}

// isOptionalMessage reports whether this is an optional message.
// If the kind is unknown, it is assumed to be a message.
func isOptionalMessage(fd protoreflect.FieldDescriptor) bool {
	return (fd.Kind() == 0 || fd.Kind() == protoreflect.MessageKind) && fd.Cardinality() == protoreflect.Optional
}

// isPackable checks whether the pack option can be specified.
func isPackable(fd protoreflect.FieldDescriptor) bool {
	switch fd.Kind() {
	case protoreflect.StringKind, protoreflect.BytesKind, protoreflect.MessageKind, protoreflect.GroupKind:
		return false
	}
	return fd.IsList()
}

// checkValidGroup reports whether fd is a valid group according to the same
// rules that protoc imposes.
func checkValidGroup(fd protoreflect.FieldDescriptor) error {
	md := fd.Message()
	switch {
	case fd.Kind() != protoreflect.GroupKind:
		return nil
	case fd.Syntax() == protoreflect.Proto3:
		return errors.New("invalid under proto3 semantics")
	case md == nil || md.IsPlaceholder():
		return errors.New("message must be resolvable")
	case fd.FullName().Parent() != md.FullName().Parent():
		return errors.New("message and field must be declared in the same scope")
	case !unicode.IsUpper(rune(md.Name()[0])):
		return errors.New("message name must start with an uppercase")
	case fd.Name() != protoreflect.Name(strings.ToLower(string(md.Name()))):
		return errors.New("field name must be lowercased form of the message name")
	}
	return nil
}

// checkValidMap checks whether the field is a valid map according to the same
// rules that protoc imposes.
// See protoc v3.8.0: src/google/protobuf/descriptor.cc:6045-6115
func checkValidMap(fd protoreflect.FieldDescriptor) error {
	md := fd.Message()
	switch {
	case md == nil || !md.IsMapEntry():
		return nil
	case fd.FullName().Parent() != md.FullName().Parent():
		return errors.New("message and field must be declared in the same scope")
	case md.Name() != protoreflect.Name(strs.MapEntryName(string(fd.Name()))):
		return errors.New("incorrect implicit map entry name")
	case fd.Cardinality() != protoreflect.Repeated:
		return errors.New("field must be repeated")
	case md.Fields().Len() != 2:
		return errors.New("message must have exactly two fields")
	case md.ExtensionRanges().Len() > 0:
		return errors.New("message must not have any extension ranges")
	case md.Enums().Len()+md.Messages().Len()+md.Extensions().Len() > 0:
		return errors.New("message must not have any nested declarations")
	}
	kf := md.Fields().Get(0)
	vf := md.Fields().Get(1)
	switch {
	case kf.Name() != genid.MapEntry_Key_field_name || kf.Number() != genid.MapEntry_Key_field_number || kf.Cardinality() != protoreflect.Optional || kf.ContainingOneof() != nil || kf.HasDefault():
		return errors.New("invalid key field")
	case vf.Name() != genid.MapEntry_Value_field_name || vf.Number() != genid.MapEntry_Value_field_number || vf.Cardinality() != protoreflect.Optional || vf.ContainingOneof() != nil || vf.HasDefault():
		return errors.New("invalid value field")
	}
	switch kf.Kind() {
	case protoreflect.BoolKind: // bool
	case protoreflect.Int32Kind, protoreflect.Sint32Kind, protoreflect.Sfixed32Kind: // int32
	case protoreflect.Int64Kind, protoreflect.Sint64Kind, protoreflect.Sfixed64Kind: // int64
	case protoreflect.Uint32Kind, protoreflect.Fixed32Kind: // uint32
	case protoreflect.Uint64Kind, protoreflect.Fixed64Kind: // uint64
	case protoreflect.StringKind: // string
	default:
		return errors.New("invalid key kind: %v", kf.Kind())
	}
	if e := vf.Enum(); e != nil && e.Values().Len() > 0 && e.Values().Get(0).Number() != 0 {
		return errors.New("map enum value must have zero number for the first value")
	}
	return nil
}
