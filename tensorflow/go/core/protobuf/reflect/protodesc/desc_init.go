// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

type descsByName map[protoreflect.FullName]protoreflect.Descriptor

func (r descsByName) initEnumDeclarations(eds []*descriptorpb.EnumDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (es []filedesc.Enum, err error) {
	es = make([]filedesc.Enum, len(eds)) // allocate up-front to ensure stable pointers
	for i, ed := range eds {
		e := &es[i]
		e.L2 = new(filedesc.EnumL2)
		if e.L0, err = r.makeBase(e, parent, ed.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := ed.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.EnumOptions)
			e.L2.Options = func() protoreflect.ProtoMessage { return opts }
		}
		e.L1.EditionFeatures = mergeEditionFeatures(parent, ed.GetOptions().GetFeatures())
		for _, s := range ed.GetReservedName() {
			e.L2.ReservedNames.List = append(e.L2.ReservedNames.List, protoreflect.Name(s))
		}
		for _, rr := range ed.GetReservedRange() {
			e.L2.ReservedRanges.List = append(e.L2.ReservedRanges.List, [2]protoreflect.EnumNumber{
				protoreflect.EnumNumber(rr.GetStart()),
				protoreflect.EnumNumber(rr.GetEnd()),
			})
		}
		if e.L2.Values.List, err = r.initEnumValuesFromDescriptorProto(ed.GetValue(), e, sb); err != nil {
			return nil, err
		}
	}
	return es, nil
}

func (r descsByName) initEnumValuesFromDescriptorProto(vds []*descriptorpb.EnumValueDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (vs []filedesc.EnumValue, err error) {
	vs = make([]filedesc.EnumValue, len(vds)) // allocate up-front to ensure stable pointers
	for i, vd := range vds {
		v := &vs[i]
		if v.L0, err = r.makeBase(v, parent, vd.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := vd.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.EnumValueOptions)
			v.L1.Options = func() protoreflect.ProtoMessage { return opts }
		}
		v.L1.Number = protoreflect.EnumNumber(vd.GetNumber())
	}
	return vs, nil
}

func (r descsByName) initMessagesDeclarations(mds []*descriptorpb.DescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (ms []filedesc.Message, err error) {
	ms = make([]filedesc.Message, len(mds)) // allocate up-front to ensure stable pointers
	for i, md := range mds {
		m := &ms[i]
		m.L2 = new(filedesc.MessageL2)
		if m.L0, err = r.makeBase(m, parent, md.GetName(), i, sb); err != nil {
			return nil, err
		}
		if m.Base.L0.ParentFile.Syntax() == protoreflect.Editions {
			m.L1.EditionFeatures = mergeEditionFeatures(parent, md.GetOptions().GetFeatures())
		}
		if opts := md.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.MessageOptions)
			m.L2.Options = func() protoreflect.ProtoMessage { return opts }
			m.L1.IsMapEntry = opts.GetMapEntry()
			m.L1.IsMessageSet = opts.GetMessageSetWireFormat()
		}
		for _, s := range md.GetReservedName() {
			m.L2.ReservedNames.List = append(m.L2.ReservedNames.List, protoreflect.Name(s))
		}
		for _, rr := range md.GetReservedRange() {
			m.L2.ReservedRanges.List = append(m.L2.ReservedRanges.List, [2]protoreflect.FieldNumber{
				protoreflect.FieldNumber(rr.GetStart()),
				protoreflect.FieldNumber(rr.GetEnd()),
			})
		}
		for _, xr := range md.GetExtensionRange() {
			m.L2.ExtensionRanges.List = append(m.L2.ExtensionRanges.List, [2]protoreflect.FieldNumber{
				protoreflect.FieldNumber(xr.GetStart()),
				protoreflect.FieldNumber(xr.GetEnd()),
			})
			var optsFunc func() protoreflect.ProtoMessage
			if opts := xr.GetOptions(); opts != nil {
				opts = proto.Clone(opts).(*descriptorpb.ExtensionRangeOptions)
				optsFunc = func() protoreflect.ProtoMessage { return opts }
			}
			m.L2.ExtensionRangeOptions = append(m.L2.ExtensionRangeOptions, optsFunc)
		}
		if m.L2.Fields.List, err = r.initFieldsFromDescriptorProto(md.GetField(), m, sb); err != nil {
			return nil, err
		}
		if m.L2.Oneofs.List, err = r.initOneofsFromDescriptorProto(md.GetOneofDecl(), m, sb); err != nil {
			return nil, err
		}
		if m.L1.Enums.List, err = r.initEnumDeclarations(md.GetEnumType(), m, sb); err != nil {
			return nil, err
		}
		if m.L1.Messages.List, err = r.initMessagesDeclarations(md.GetNestedType(), m, sb); err != nil {
			return nil, err
		}
		if m.L1.Extensions.List, err = r.initExtensionDeclarations(md.GetExtension(), m, sb); err != nil {
			return nil, err
		}
	}
	return ms, nil
}

// canBePacked returns whether the field can use packed encoding:
// https://protobuf.dev/programming-guides/encoding/#packed
func canBePacked(fd *descriptorpb.FieldDescriptorProto) bool {
	if fd.GetLabel() != descriptorpb.FieldDescriptorProto_LABEL_REPEATED {
		return false // not a repeated field
	}

	switch protoreflect.Kind(fd.GetType()) {
	case protoreflect.MessageKind, protoreflect.GroupKind:
		return false // not a scalar type field

	case protoreflect.StringKind, protoreflect.BytesKind:
		// string and bytes can explicitly not be declared as packed,
		// see https://protobuf.dev/programming-guides/encoding/#packed
		return false

	default:
		return true
	}
}

func (r descsByName) initFieldsFromDescriptorProto(fds []*descriptorpb.FieldDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (fs []filedesc.Field, err error) {
	fs = make([]filedesc.Field, len(fds)) // allocate up-front to ensure stable pointers
	for i, fd := range fds {
		f := &fs[i]
		if f.L0, err = r.makeBase(f, parent, fd.GetName(), i, sb); err != nil {
			return nil, err
		}
		f.L1.IsProto3Optional = fd.GetProto3Optional()
		if opts := fd.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.FieldOptions)
			f.L1.Options = func() protoreflect.ProtoMessage { return opts }
			f.L1.IsWeak = opts.GetWeak()
			f.L1.HasPacked = opts.Packed != nil
			f.L1.IsPacked = opts.GetPacked()
		}
		f.L1.Number = protoreflect.FieldNumber(fd.GetNumber())
		f.L1.Cardinality = protoreflect.Cardinality(fd.GetLabel())
		if fd.Type != nil {
			f.L1.Kind = protoreflect.Kind(fd.GetType())
		}
		if fd.JsonName != nil {
			f.L1.StringName.InitJSON(fd.GetJsonName())
		}

		if f.Base.L0.ParentFile.Syntax() == protoreflect.Editions {
			f.L1.EditionFeatures = mergeEditionFeatures(parent, fd.GetOptions().GetFeatures())

			if f.L1.EditionFeatures.IsLegacyRequired {
				f.L1.Cardinality = protoreflect.Required
			}
			// We reuse the existing field because the old option `[packed =
			// true]` is mutually exclusive with the editions feature.
			if canBePacked(fd) {
				f.L1.HasPacked = true
				f.L1.IsPacked = f.L1.EditionFeatures.IsPacked
			}

			// We pretend this option is always explicitly set because the only
			// use of HasEnforceUTF8 is to determine whether to use EnforceUTF8
			// or to return the appropriate default.
			// When using editions we either parse the option or resolve the
			// appropriate default here (instead of later when this option is
			// requested from the descriptor).
			// In proto2/proto3 syntax HasEnforceUTF8 might be false.
			f.L1.HasEnforceUTF8 = true
			f.L1.EnforceUTF8 = f.L1.EditionFeatures.IsUTF8Validated

			if f.L1.Kind == protoreflect.MessageKind && f.L1.EditionFeatures.IsDelimitedEncoded {
				f.L1.Kind = protoreflect.GroupKind
			}
		}
	}
	return fs, nil
}

func (r descsByName) initOneofsFromDescriptorProto(ods []*descriptorpb.OneofDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (os []filedesc.Oneof, err error) {
	os = make([]filedesc.Oneof, len(ods)) // allocate up-front to ensure stable pointers
	for i, od := range ods {
		o := &os[i]
		if o.L0, err = r.makeBase(o, parent, od.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := od.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.OneofOptions)
			o.L1.Options = func() protoreflect.ProtoMessage { return opts }
			if parent.Syntax() == protoreflect.Editions {
				o.L1.EditionFeatures = mergeEditionFeatures(parent, opts.GetFeatures())
			}
		}
	}
	return os, nil
}

func (r descsByName) initExtensionDeclarations(xds []*descriptorpb.FieldDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (xs []filedesc.Extension, err error) {
	xs = make([]filedesc.Extension, len(xds)) // allocate up-front to ensure stable pointers
	for i, xd := range xds {
		x := &xs[i]
		x.L2 = new(filedesc.ExtensionL2)
		if x.L0, err = r.makeBase(x, parent, xd.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := xd.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.FieldOptions)
			x.L2.Options = func() protoreflect.ProtoMessage { return opts }
			x.L2.IsPacked = opts.GetPacked()
		}
		x.L1.Number = protoreflect.FieldNumber(xd.GetNumber())
		x.L1.Cardinality = protoreflect.Cardinality(xd.GetLabel())
		if xd.Type != nil {
			x.L1.Kind = protoreflect.Kind(xd.GetType())
		}
		if xd.JsonName != nil {
			x.L2.StringName.InitJSON(xd.GetJsonName())
		}
	}
	return xs, nil
}

func (r descsByName) initServiceDeclarations(sds []*descriptorpb.ServiceDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (ss []filedesc.Service, err error) {
	ss = make([]filedesc.Service, len(sds)) // allocate up-front to ensure stable pointers
	for i, sd := range sds {
		s := &ss[i]
		s.L2 = new(filedesc.ServiceL2)
		if s.L0, err = r.makeBase(s, parent, sd.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := sd.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.ServiceOptions)
			s.L2.Options = func() protoreflect.ProtoMessage { return opts }
		}
		if s.L2.Methods.List, err = r.initMethodsFromDescriptorProto(sd.GetMethod(), s, sb); err != nil {
			return nil, err
		}
	}
	return ss, nil
}

func (r descsByName) initMethodsFromDescriptorProto(mds []*descriptorpb.MethodDescriptorProto, parent protoreflect.Descriptor, sb *strs.Builder) (ms []filedesc.Method, err error) {
	ms = make([]filedesc.Method, len(mds)) // allocate up-front to ensure stable pointers
	for i, md := range mds {
		m := &ms[i]
		if m.L0, err = r.makeBase(m, parent, md.GetName(), i, sb); err != nil {
			return nil, err
		}
		if opts := md.GetOptions(); opts != nil {
			opts = proto.Clone(opts).(*descriptorpb.MethodOptions)
			m.L1.Options = func() protoreflect.ProtoMessage { return opts }
		}
		m.L1.IsStreamingClient = md.GetClientStreaming()
		m.L1.IsStreamingServer = md.GetServerStreaming()
	}
	return ms, nil
}

func (r descsByName) makeBase(child, parent protoreflect.Descriptor, name string, idx int, sb *strs.Builder) (filedesc.BaseL0, error) {
	if !protoreflect.Name(name).IsValid() {
		return filedesc.BaseL0{}, errors.New("descriptor %q has an invalid nested name: %q", parent.FullName(), name)
	}

	// Derive the full name of the child.
	// Note that enum values are a sibling to the enum parent in the namespace.
	var fullName protoreflect.FullName
	if _, ok := parent.(protoreflect.EnumDescriptor); ok {
		fullName = sb.AppendFullName(parent.FullName().Parent(), protoreflect.Name(name))
	} else {
		fullName = sb.AppendFullName(parent.FullName(), protoreflect.Name(name))
	}
	if _, ok := r[fullName]; ok {
		return filedesc.BaseL0{}, errors.New("descriptor %q already declared", fullName)
	}
	r[fullName] = child

	// TODO: Verify that the full name does not already exist in the resolver?
	// This is not as critical since most usages of NewFile will register
	// the created file back into the registry, which will perform this check.

	return filedesc.BaseL0{
		FullName:   fullName,
		ParentFile: parent.ParentFile().(*filedesc.File),
		Parent:     parent,
		Index:      idx,
	}, nil
}
