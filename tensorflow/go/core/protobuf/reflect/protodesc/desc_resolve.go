// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"google.golang.org/protobuf/internal/encoding/defval"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	"google.golang.org/protobuf/types/descriptorpb"
)

// resolver is a wrapper around a local registry of declarations within the file
// and the remote resolver. The remote resolver is restricted to only return
// descriptors that have been imported.
type resolver struct {
	local   descsByName
	remote  Resolver
	imports importSet

	allowUnresolvable bool
}

func (r *resolver) resolveMessageDependencies(ms []filedesc.Message, mds []*descriptorpb.DescriptorProto) (err error) {
	for i, md := range mds {
		m := &ms[i]
		for j, fd := range md.GetField() {
			f := &m.L2.Fields.List[j]
			if f.L1.Cardinality == protoreflect.Required {
				m.L2.RequiredNumbers.List = append(m.L2.RequiredNumbers.List, f.L1.Number)
			}
			if fd.OneofIndex != nil {
				k := int(fd.GetOneofIndex())
				if !(0 <= k && k < len(md.GetOneofDecl())) {
					return errors.New("message field %q has an invalid oneof index: %d", f.FullName(), k)
				}
				o := &m.L2.Oneofs.List[k]
				f.L1.ContainingOneof = o
				o.L1.Fields.List = append(o.L1.Fields.List, f)
			}

			if f.L1.Kind, f.L1.Enum, f.L1.Message, err = r.findTarget(f.Kind(), f.Parent().FullName(), partialName(fd.GetTypeName()), f.IsWeak()); err != nil {
				return errors.New("message field %q cannot resolve type: %v", f.FullName(), err)
			}
			if fd.DefaultValue != nil {
				v, ev, err := unmarshalDefault(fd.GetDefaultValue(), f, r.allowUnresolvable)
				if err != nil {
					return errors.New("message field %q has invalid default: %v", f.FullName(), err)
				}
				f.L1.Default = filedesc.DefaultValue(v, ev)
			}
		}

		if err := r.resolveMessageDependencies(m.L1.Messages.List, md.GetNestedType()); err != nil {
			return err
		}
		if err := r.resolveExtensionDependencies(m.L1.Extensions.List, md.GetExtension()); err != nil {
			return err
		}
	}
	return nil
}

func (r *resolver) resolveExtensionDependencies(xs []filedesc.Extension, xds []*descriptorpb.FieldDescriptorProto) (err error) {
	for i, xd := range xds {
		x := &xs[i]
		if x.L1.Extendee, err = r.findMessageDescriptor(x.Parent().FullName(), partialName(xd.GetExtendee()), false); err != nil {
			return errors.New("extension field %q cannot resolve extendee: %v", x.FullName(), err)
		}
		if x.L1.Kind, x.L2.Enum, x.L2.Message, err = r.findTarget(x.Kind(), x.Parent().FullName(), partialName(xd.GetTypeName()), false); err != nil {
			return errors.New("extension field %q cannot resolve type: %v", x.FullName(), err)
		}
		if xd.DefaultValue != nil {
			v, ev, err := unmarshalDefault(xd.GetDefaultValue(), x, r.allowUnresolvable)
			if err != nil {
				return errors.New("extension field %q has invalid default: %v", x.FullName(), err)
			}
			x.L2.Default = filedesc.DefaultValue(v, ev)
		}
	}
	return nil
}

func (r *resolver) resolveServiceDependencies(ss []filedesc.Service, sds []*descriptorpb.ServiceDescriptorProto) (err error) {
	for i, sd := range sds {
		s := &ss[i]
		for j, md := range sd.GetMethod() {
			m := &s.L2.Methods.List[j]
			m.L1.Input, err = r.findMessageDescriptor(m.Parent().FullName(), partialName(md.GetInputType()), false)
			if err != nil {
				return errors.New("service method %q cannot resolve input: %v", m.FullName(), err)
			}
			m.L1.Output, err = r.findMessageDescriptor(s.FullName(), partialName(md.GetOutputType()), false)
			if err != nil {
				return errors.New("service method %q cannot resolve output: %v", m.FullName(), err)
			}
		}
	}
	return nil
}

// findTarget finds an enum or message descriptor if k is an enum, message,
// group, or unknown. If unknown, and the name could be resolved, the kind
// returned kind is set based on the type of the resolved descriptor.
func (r *resolver) findTarget(k protoreflect.Kind, scope protoreflect.FullName, ref partialName, isWeak bool) (protoreflect.Kind, protoreflect.EnumDescriptor, protoreflect.MessageDescriptor, error) {
	switch k {
	case protoreflect.EnumKind:
		ed, err := r.findEnumDescriptor(scope, ref, isWeak)
		if err != nil {
			return 0, nil, nil, err
		}
		return k, ed, nil, nil
	case protoreflect.MessageKind, protoreflect.GroupKind:
		md, err := r.findMessageDescriptor(scope, ref, isWeak)
		if err != nil {
			return 0, nil, nil, err
		}
		return k, nil, md, nil
	case 0:
		// Handle unspecified kinds (possible with parsers that operate
		// on a per-file basis without knowledge of dependencies).
		d, err := r.findDescriptor(scope, ref)
		if err == protoregistry.NotFound && (r.allowUnresolvable || isWeak) {
			return k, filedesc.PlaceholderEnum(ref.FullName()), filedesc.PlaceholderMessage(ref.FullName()), nil
		} else if err == protoregistry.NotFound {
			return 0, nil, nil, errors.New("%q not found", ref.FullName())
		} else if err != nil {
			return 0, nil, nil, err
		}
		switch d := d.(type) {
		case protoreflect.EnumDescriptor:
			return protoreflect.EnumKind, d, nil, nil
		case protoreflect.MessageDescriptor:
			return protoreflect.MessageKind, nil, d, nil
		default:
			return 0, nil, nil, errors.New("unknown kind")
		}
	default:
		if ref != "" {
			return 0, nil, nil, errors.New("target name cannot be specified for %v", k)
		}
		if !k.IsValid() {
			return 0, nil, nil, errors.New("invalid kind: %d", k)
		}
		return k, nil, nil, nil
	}
}

// findDescriptor finds the descriptor by name,
// which may be a relative name within some scope.
//
// Suppose the scope was "fizz.buzz" and the reference was "Foo.Bar",
// then the following full names are searched:
//   - fizz.buzz.Foo.Bar
//   - fizz.Foo.Bar
//   - Foo.Bar
func (r *resolver) findDescriptor(scope protoreflect.FullName, ref partialName) (protoreflect.Descriptor, error) {
	if !ref.IsValid() {
		return nil, errors.New("invalid name reference: %q", ref)
	}
	if ref.IsFull() {
		scope, ref = "", ref[1:]
	}
	var foundButNotImported protoreflect.Descriptor
	for {
		// Derive the full name to search.
		s := protoreflect.FullName(ref)
		if scope != "" {
			s = scope + "." + s
		}

		// Check the current file for the descriptor.
		if d, ok := r.local[s]; ok {
			return d, nil
		}

		// Check the remote registry for the descriptor.
		d, err := r.remote.FindDescriptorByName(s)
		if err == nil {
			// Only allow descriptors covered by one of the imports.
			if r.imports[d.ParentFile().Path()] {
				return d, nil
			}
			foundButNotImported = d
		} else if err != protoregistry.NotFound {
			return nil, errors.Wrap(err, "%q", s)
		}

		// Continue on at a higher level of scoping.
		if scope == "" {
			if d := foundButNotImported; d != nil {
				return nil, errors.New("resolved %q, but %q is not imported", d.FullName(), d.ParentFile().Path())
			}
			return nil, protoregistry.NotFound
		}
		scope = scope.Parent()
	}
}

func (r *resolver) findEnumDescriptor(scope protoreflect.FullName, ref partialName, isWeak bool) (protoreflect.EnumDescriptor, error) {
	d, err := r.findDescriptor(scope, ref)
	if err == protoregistry.NotFound && (r.allowUnresolvable || isWeak) {
		return filedesc.PlaceholderEnum(ref.FullName()), nil
	} else if err == protoregistry.NotFound {
		return nil, errors.New("%q not found", ref.FullName())
	} else if err != nil {
		return nil, err
	}
	ed, ok := d.(protoreflect.EnumDescriptor)
	if !ok {
		return nil, errors.New("resolved %q, but it is not an enum", d.FullName())
	}
	return ed, nil
}

func (r *resolver) findMessageDescriptor(scope protoreflect.FullName, ref partialName, isWeak bool) (protoreflect.MessageDescriptor, error) {
	d, err := r.findDescriptor(scope, ref)
	if err == protoregistry.NotFound && (r.allowUnresolvable || isWeak) {
		return filedesc.PlaceholderMessage(ref.FullName()), nil
	} else if err == protoregistry.NotFound {
		return nil, errors.New("%q not found", ref.FullName())
	} else if err != nil {
		return nil, err
	}
	md, ok := d.(protoreflect.MessageDescriptor)
	if !ok {
		return nil, errors.New("resolved %q, but it is not an message", d.FullName())
	}
	return md, nil
}

// partialName is the partial name. A leading dot means that the name is full,
// otherwise the name is relative to some current scope.
// See google.protobuf.FieldDescriptorProto.type_name.
type partialName string

func (s partialName) IsFull() bool {
	return len(s) > 0 && s[0] == '.'
}

func (s partialName) IsValid() bool {
	if s.IsFull() {
		return protoreflect.FullName(s[1:]).IsValid()
	}
	return protoreflect.FullName(s).IsValid()
}

const unknownPrefix = "*."

// FullName converts the partial name to a full name on a best-effort basis.
// If relative, it creates an invalid full name, using a "*." prefix
// to indicate that the start of the full name is unknown.
func (s partialName) FullName() protoreflect.FullName {
	if s.IsFull() {
		return protoreflect.FullName(s[1:])
	}
	return protoreflect.FullName(unknownPrefix + s)
}

func unmarshalDefault(s string, fd protoreflect.FieldDescriptor, allowUnresolvable bool) (protoreflect.Value, protoreflect.EnumValueDescriptor, error) {
	var evs protoreflect.EnumValueDescriptors
	if fd.Enum() != nil {
		evs = fd.Enum().Values()
	}
	v, ev, err := defval.Unmarshal(s, fd.Kind(), evs, defval.Descriptor)
	if err != nil && allowUnresolvable && evs != nil && protoreflect.Name(s).IsValid() {
		v = protoreflect.ValueOfEnum(0)
		if evs.Len() > 0 {
			v = protoreflect.ValueOfEnum(evs.Get(0).Number())
		}
		ev = filedesc.PlaceholderEnumValue(fd.Enum().FullName().Parent().Append(protoreflect.Name(s)))
	} else if err != nil {
		return v, ev, err
	}
	if !fd.HasPresence() {
		return v, ev, errors.New("cannot be specified with implicit field presence")
	}
	if fd.Kind() == protoreflect.MessageKind || fd.Kind() == protoreflect.GroupKind || fd.Cardinality() == protoreflect.Repeated {
		return v, ev, errors.New("cannot be specified on composite types")
	}
	return v, ev, nil
}
