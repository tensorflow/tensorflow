// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"reflect"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/descopts"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func (fd *File) lazyRawInit() {
	fd.unmarshalFull(fd.builder.RawDescriptor)
	fd.resolveMessages()
	fd.resolveExtensions()
	fd.resolveServices()
}

func (file *File) resolveMessages() {
	var depIdx int32
	for i := range file.allMessages {
		md := &file.allMessages[i]

		// Resolve message field dependencies.
		for j := range md.L2.Fields.List {
			fd := &md.L2.Fields.List[j]

			// Weak fields are resolved upon actual use.
			if fd.L1.IsWeak {
				continue
			}

			// Resolve message field dependency.
			switch fd.L1.Kind {
			case protoreflect.EnumKind:
				fd.L1.Enum = file.resolveEnumDependency(fd.L1.Enum, listFieldDeps, depIdx)
				depIdx++
			case protoreflect.MessageKind, protoreflect.GroupKind:
				fd.L1.Message = file.resolveMessageDependency(fd.L1.Message, listFieldDeps, depIdx)
				depIdx++
			}

			// Default is resolved here since it depends on Enum being resolved.
			if v := fd.L1.Default.val; v.IsValid() {
				fd.L1.Default = unmarshalDefault(v.Bytes(), fd.L1.Kind, file, fd.L1.Enum)
			}
		}
	}
}

func (file *File) resolveExtensions() {
	var depIdx int32
	for i := range file.allExtensions {
		xd := &file.allExtensions[i]

		// Resolve extension field dependency.
		switch xd.L1.Kind {
		case protoreflect.EnumKind:
			xd.L2.Enum = file.resolveEnumDependency(xd.L2.Enum, listExtDeps, depIdx)
			depIdx++
		case protoreflect.MessageKind, protoreflect.GroupKind:
			xd.L2.Message = file.resolveMessageDependency(xd.L2.Message, listExtDeps, depIdx)
			depIdx++
		}

		// Default is resolved here since it depends on Enum being resolved.
		if v := xd.L2.Default.val; v.IsValid() {
			xd.L2.Default = unmarshalDefault(v.Bytes(), xd.L1.Kind, file, xd.L2.Enum)
		}
	}
}

func (file *File) resolveServices() {
	var depIdx int32
	for i := range file.allServices {
		sd := &file.allServices[i]

		// Resolve method dependencies.
		for j := range sd.L2.Methods.List {
			md := &sd.L2.Methods.List[j]
			md.L1.Input = file.resolveMessageDependency(md.L1.Input, listMethInDeps, depIdx)
			md.L1.Output = file.resolveMessageDependency(md.L1.Output, listMethOutDeps, depIdx)
			depIdx++
		}
	}
}

func (file *File) resolveEnumDependency(ed protoreflect.EnumDescriptor, i, j int32) protoreflect.EnumDescriptor {
	r := file.builder.FileRegistry
	if r, ok := r.(resolverByIndex); ok {
		if ed2 := r.FindEnumByIndex(i, j, file.allEnums, file.allMessages); ed2 != nil {
			return ed2
		}
	}
	for i := range file.allEnums {
		if ed2 := &file.allEnums[i]; ed2.L0.FullName == ed.FullName() {
			return ed2
		}
	}
	if d, _ := r.FindDescriptorByName(ed.FullName()); d != nil {
		return d.(protoreflect.EnumDescriptor)
	}
	return ed
}

func (file *File) resolveMessageDependency(md protoreflect.MessageDescriptor, i, j int32) protoreflect.MessageDescriptor {
	r := file.builder.FileRegistry
	if r, ok := r.(resolverByIndex); ok {
		if md2 := r.FindMessageByIndex(i, j, file.allEnums, file.allMessages); md2 != nil {
			return md2
		}
	}
	for i := range file.allMessages {
		if md2 := &file.allMessages[i]; md2.L0.FullName == md.FullName() {
			return md2
		}
	}
	if d, _ := r.FindDescriptorByName(md.FullName()); d != nil {
		return d.(protoreflect.MessageDescriptor)
	}
	return md
}

func (fd *File) unmarshalFull(b []byte) {
	sb := getBuilder()
	defer putBuilder(sb)

	var enumIdx, messageIdx, extensionIdx, serviceIdx int
	var rawOptions []byte
	fd.L2 = new(FileL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FileDescriptorProto_PublicDependency_field_number:
				fd.L2.Imports[v].IsPublic = true
			case genid.FileDescriptorProto_WeakDependency_field_number:
				fd.L2.Imports[v].IsWeak = true
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FileDescriptorProto_Dependency_field_number:
				path := sb.MakeString(v)
				imp, _ := fd.builder.FileRegistry.FindFileByPath(path)
				if imp == nil {
					imp = PlaceholderFile(path)
				}
				fd.L2.Imports = append(fd.L2.Imports, protoreflect.FileImport{FileDescriptor: imp})
			case genid.FileDescriptorProto_EnumType_field_number:
				fd.L1.Enums.List[enumIdx].unmarshalFull(v, sb)
				enumIdx++
			case genid.FileDescriptorProto_MessageType_field_number:
				fd.L1.Messages.List[messageIdx].unmarshalFull(v, sb)
				messageIdx++
			case genid.FileDescriptorProto_Extension_field_number:
				fd.L1.Extensions.List[extensionIdx].unmarshalFull(v, sb)
				extensionIdx++
			case genid.FileDescriptorProto_Service_field_number:
				fd.L1.Services.List[serviceIdx].unmarshalFull(v, sb)
				serviceIdx++
			case genid.FileDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	fd.L2.Options = fd.builder.optionsUnmarshaler(&descopts.File, rawOptions)
}

func (ed *Enum) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawValues [][]byte
	var rawOptions []byte
	if !ed.L1.eagerValues {
		ed.L2 = new(EnumL2)
	}
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.EnumDescriptorProto_Value_field_number:
				rawValues = append(rawValues, v)
			case genid.EnumDescriptorProto_ReservedName_field_number:
				ed.L2.ReservedNames.List = append(ed.L2.ReservedNames.List, protoreflect.Name(sb.MakeString(v)))
			case genid.EnumDescriptorProto_ReservedRange_field_number:
				ed.L2.ReservedRanges.List = append(ed.L2.ReservedRanges.List, unmarshalEnumReservedRange(v))
			case genid.EnumDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if !ed.L1.eagerValues && len(rawValues) > 0 {
		ed.L2.Values.List = make([]EnumValue, len(rawValues))
		for i, b := range rawValues {
			ed.L2.Values.List[i].unmarshalFull(b, sb, ed.L0.ParentFile, ed, i)
		}
	}
	ed.L2.Options = ed.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Enum, rawOptions)
}

func unmarshalEnumReservedRange(b []byte) (r [2]protoreflect.EnumNumber) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.EnumDescriptorProto_EnumReservedRange_Start_field_number:
				r[0] = protoreflect.EnumNumber(v)
			case genid.EnumDescriptorProto_EnumReservedRange_End_field_number:
				r[1] = protoreflect.EnumNumber(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r
}

func (vd *EnumValue) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	vd.L0.ParentFile = pf
	vd.L0.Parent = pd
	vd.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.EnumValueDescriptorProto_Number_field_number:
				vd.L1.Number = protoreflect.EnumNumber(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.EnumValueDescriptorProto_Name_field_number:
				// NOTE: Enum values are in the same scope as the enum parent.
				vd.L0.FullName = appendFullName(sb, pd.Parent().FullName(), v)
			case genid.EnumValueDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	vd.L1.Options = pf.builder.optionsUnmarshaler(&descopts.EnumValue, rawOptions)
}

func (md *Message) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawFields, rawOneofs [][]byte
	var enumIdx, messageIdx, extensionIdx int
	var rawOptions []byte
	md.L1.EditionFeatures = featuresFromParentDesc(md.Parent())
	md.L2 = new(MessageL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.DescriptorProto_Field_field_number:
				rawFields = append(rawFields, v)
			case genid.DescriptorProto_OneofDecl_field_number:
				rawOneofs = append(rawOneofs, v)
			case genid.DescriptorProto_ReservedName_field_number:
				md.L2.ReservedNames.List = append(md.L2.ReservedNames.List, protoreflect.Name(sb.MakeString(v)))
			case genid.DescriptorProto_ReservedRange_field_number:
				md.L2.ReservedRanges.List = append(md.L2.ReservedRanges.List, unmarshalMessageReservedRange(v))
			case genid.DescriptorProto_ExtensionRange_field_number:
				r, rawOptions := unmarshalMessageExtensionRange(v)
				opts := md.L0.ParentFile.builder.optionsUnmarshaler(&descopts.ExtensionRange, rawOptions)
				md.L2.ExtensionRanges.List = append(md.L2.ExtensionRanges.List, r)
				md.L2.ExtensionRangeOptions = append(md.L2.ExtensionRangeOptions, opts)
			case genid.DescriptorProto_EnumType_field_number:
				md.L1.Enums.List[enumIdx].unmarshalFull(v, sb)
				enumIdx++
			case genid.DescriptorProto_NestedType_field_number:
				md.L1.Messages.List[messageIdx].unmarshalFull(v, sb)
				messageIdx++
			case genid.DescriptorProto_Extension_field_number:
				md.L1.Extensions.List[extensionIdx].unmarshalFull(v, sb)
				extensionIdx++
			case genid.DescriptorProto_Options_field_number:
				md.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if len(rawFields) > 0 || len(rawOneofs) > 0 {
		md.L2.Fields.List = make([]Field, len(rawFields))
		md.L2.Oneofs.List = make([]Oneof, len(rawOneofs))
		for i, b := range rawFields {
			fd := &md.L2.Fields.List[i]
			fd.unmarshalFull(b, sb, md.L0.ParentFile, md, i)
			if fd.L1.Cardinality == protoreflect.Required {
				md.L2.RequiredNumbers.List = append(md.L2.RequiredNumbers.List, fd.L1.Number)
			}
		}
		for i, b := range rawOneofs {
			od := &md.L2.Oneofs.List[i]
			od.unmarshalFull(b, sb, md.L0.ParentFile, md, i)
		}
	}
	md.L2.Options = md.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Message, rawOptions)
}

func (md *Message) unmarshalOptions(b []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.MessageOptions_MapEntry_field_number:
				md.L1.IsMapEntry = protowire.DecodeBool(v)
			case genid.MessageOptions_MessageSetWireFormat_field_number:
				md.L1.IsMessageSet = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.MessageOptions_Features_field_number:
				md.L1.EditionFeatures = unmarshalFeatureSet(v, md.L1.EditionFeatures)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func unmarshalMessageReservedRange(b []byte) (r [2]protoreflect.FieldNumber) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.DescriptorProto_ReservedRange_Start_field_number:
				r[0] = protoreflect.FieldNumber(v)
			case genid.DescriptorProto_ReservedRange_End_field_number:
				r[1] = protoreflect.FieldNumber(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r
}

func unmarshalMessageExtensionRange(b []byte) (r [2]protoreflect.FieldNumber, rawOptions []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.DescriptorProto_ExtensionRange_Start_field_number:
				r[0] = protoreflect.FieldNumber(v)
			case genid.DescriptorProto_ExtensionRange_End_field_number:
				r[1] = protoreflect.FieldNumber(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.DescriptorProto_ExtensionRange_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	return r, rawOptions
}

func (fd *Field) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	fd.L0.ParentFile = pf
	fd.L0.Parent = pd
	fd.L0.Index = i
	fd.L1.EditionFeatures = featuresFromParentDesc(fd.Parent())

	var rawTypeName []byte
	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_Number_field_number:
				fd.L1.Number = protoreflect.FieldNumber(v)
			case genid.FieldDescriptorProto_Label_field_number:
				fd.L1.Cardinality = protoreflect.Cardinality(v)
			case genid.FieldDescriptorProto_Type_field_number:
				fd.L1.Kind = protoreflect.Kind(v)
			case genid.FieldDescriptorProto_OneofIndex_field_number:
				// In Message.unmarshalFull, we allocate slices for both
				// the field and oneof descriptors before unmarshaling either
				// of them. This ensures pointers to slice elements are stable.
				od := &pd.(*Message).L2.Oneofs.List[v]
				od.L1.Fields.List = append(od.L1.Fields.List, fd)
				if fd.L1.ContainingOneof != nil {
					panic("oneof type already set")
				}
				fd.L1.ContainingOneof = od
			case genid.FieldDescriptorProto_Proto3Optional_field_number:
				fd.L1.IsProto3Optional = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_Name_field_number:
				fd.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.FieldDescriptorProto_JsonName_field_number:
				fd.L1.StringName.InitJSON(sb.MakeString(v))
			case genid.FieldDescriptorProto_DefaultValue_field_number:
				fd.L1.Default.val = protoreflect.ValueOfBytes(v) // temporarily store as bytes; later resolved in resolveMessages
			case genid.FieldDescriptorProto_TypeName_field_number:
				rawTypeName = v
			case genid.FieldDescriptorProto_Options_field_number:
				fd.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if fd.Syntax() == protoreflect.Editions && fd.L1.Kind == protoreflect.MessageKind && fd.L1.EditionFeatures.IsDelimitedEncoded {
		fd.L1.Kind = protoreflect.GroupKind
	}
	if fd.Syntax() == protoreflect.Editions && fd.L1.EditionFeatures.IsLegacyRequired {
		fd.L1.Cardinality = protoreflect.Required
	}
	if rawTypeName != nil {
		name := makeFullName(sb, rawTypeName)
		switch fd.L1.Kind {
		case protoreflect.EnumKind:
			fd.L1.Enum = PlaceholderEnum(name)
		case protoreflect.MessageKind, protoreflect.GroupKind:
			fd.L1.Message = PlaceholderMessage(name)
		}
	}
	fd.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Field, rawOptions)
}

func (fd *Field) unmarshalOptions(b []byte) {
	const FieldOptions_EnforceUTF8 = 13

	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FieldOptions_Packed_field_number:
				fd.L1.HasPacked = true
				fd.L1.IsPacked = protowire.DecodeBool(v)
			case genid.FieldOptions_Weak_field_number:
				fd.L1.IsWeak = protowire.DecodeBool(v)
			case FieldOptions_EnforceUTF8:
				fd.L1.HasEnforceUTF8 = true
				fd.L1.EnforceUTF8 = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FieldOptions_Features_field_number:
				fd.L1.EditionFeatures = unmarshalFeatureSet(v, fd.L1.EditionFeatures)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (od *Oneof) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	od.L0.ParentFile = pf
	od.L0.Parent = pd
	od.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.OneofDescriptorProto_Name_field_number:
				od.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.OneofDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	od.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Oneof, rawOptions)
}

func (xd *Extension) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawTypeName []byte
	var rawOptions []byte
	xd.L2 = new(ExtensionL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_Proto3Optional_field_number:
				xd.L2.IsProto3Optional = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_JsonName_field_number:
				xd.L2.StringName.InitJSON(sb.MakeString(v))
			case genid.FieldDescriptorProto_DefaultValue_field_number:
				xd.L2.Default.val = protoreflect.ValueOfBytes(v) // temporarily store as bytes; later resolved in resolveExtensions
			case genid.FieldDescriptorProto_TypeName_field_number:
				rawTypeName = v
			case genid.FieldDescriptorProto_Options_field_number:
				xd.unmarshalOptions(v)
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if rawTypeName != nil {
		name := makeFullName(sb, rawTypeName)
		switch xd.L1.Kind {
		case protoreflect.EnumKind:
			xd.L2.Enum = PlaceholderEnum(name)
		case protoreflect.MessageKind, protoreflect.GroupKind:
			xd.L2.Message = PlaceholderMessage(name)
		}
	}
	xd.L2.Options = xd.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Field, rawOptions)
}

func (xd *Extension) unmarshalOptions(b []byte) {
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FieldOptions_Packed_field_number:
				xd.L2.IsPacked = protowire.DecodeBool(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (sd *Service) unmarshalFull(b []byte, sb *strs.Builder) {
	var rawMethods [][]byte
	var rawOptions []byte
	sd.L2 = new(ServiceL2)
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.ServiceDescriptorProto_Method_field_number:
				rawMethods = append(rawMethods, v)
			case genid.ServiceDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	if len(rawMethods) > 0 {
		sd.L2.Methods.List = make([]Method, len(rawMethods))
		for i, b := range rawMethods {
			sd.L2.Methods.List[i].unmarshalFull(b, sb, sd.L0.ParentFile, sd, i)
		}
	}
	sd.L2.Options = sd.L0.ParentFile.builder.optionsUnmarshaler(&descopts.Service, rawOptions)
}

func (md *Method) unmarshalFull(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	md.L0.ParentFile = pf
	md.L0.Parent = pd
	md.L0.Index = i

	var rawOptions []byte
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.MethodDescriptorProto_ClientStreaming_field_number:
				md.L1.IsStreamingClient = protowire.DecodeBool(v)
			case genid.MethodDescriptorProto_ServerStreaming_field_number:
				md.L1.IsStreamingServer = protowire.DecodeBool(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.MethodDescriptorProto_Name_field_number:
				md.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.MethodDescriptorProto_InputType_field_number:
				md.L1.Input = PlaceholderMessage(makeFullName(sb, v))
			case genid.MethodDescriptorProto_OutputType_field_number:
				md.L1.Output = PlaceholderMessage(makeFullName(sb, v))
			case genid.MethodDescriptorProto_Options_field_number:
				rawOptions = appendOptions(rawOptions, v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
	md.L1.Options = pf.builder.optionsUnmarshaler(&descopts.Method, rawOptions)
}

// appendOptions appends src to dst, where the returned slice is never nil.
// This is necessary to distinguish between empty and unpopulated options.
func appendOptions(dst, src []byte) []byte {
	if dst == nil {
		dst = []byte{}
	}
	return append(dst, src...)
}

// optionsUnmarshaler constructs a lazy unmarshal function for an options message.
//
// The type of message to unmarshal to is passed as a pointer since the
// vars in descopts may not yet be populated at the time this function is called.
func (db *Builder) optionsUnmarshaler(p *protoreflect.ProtoMessage, b []byte) func() protoreflect.ProtoMessage {
	if b == nil {
		return nil
	}
	var opts protoreflect.ProtoMessage
	var once sync.Once
	return func() protoreflect.ProtoMessage {
		once.Do(func() {
			if *p == nil {
				panic("Descriptor.Options called without importing the descriptor package")
			}
			opts = reflect.New(reflect.TypeOf(*p).Elem()).Interface().(protoreflect.ProtoMessage)
			if err := (proto.UnmarshalOptions{
				AllowPartial: true,
				Resolver:     db.TypeResolver,
			}).Unmarshal(b, opts); err != nil {
				panic(err)
			}
		})
		return opts
	}
}
