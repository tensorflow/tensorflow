// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filedesc

import (
	"fmt"
	"sync"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// fileRaw is a data struct used when initializing a file descriptor from
// a raw FileDescriptorProto.
type fileRaw struct {
	builder       Builder
	allEnums      []Enum
	allMessages   []Message
	allExtensions []Extension
	allServices   []Service
}

func newRawFile(db Builder) *File {
	fd := &File{fileRaw: fileRaw{builder: db}}
	fd.initDecls(db.NumEnums, db.NumMessages, db.NumExtensions, db.NumServices)
	fd.unmarshalSeed(db.RawDescriptor)

	// Extended message targets are eagerly resolved since registration
	// needs this information at program init time.
	for i := range fd.allExtensions {
		xd := &fd.allExtensions[i]
		xd.L1.Extendee = fd.resolveMessageDependency(xd.L1.Extendee, listExtTargets, int32(i))
	}

	fd.checkDecls()
	return fd
}

// initDecls pre-allocates slices for the exact number of enums, messages
// (including map entries), extensions, and services declared in the proto file.
// This is done to avoid regrowing the slice, which would change the address
// for any previously seen declaration.
//
// The alloc methods "allocates" slices by pulling from the capacity.
func (fd *File) initDecls(numEnums, numMessages, numExtensions, numServices int32) {
	fd.allEnums = make([]Enum, 0, numEnums)
	fd.allMessages = make([]Message, 0, numMessages)
	fd.allExtensions = make([]Extension, 0, numExtensions)
	fd.allServices = make([]Service, 0, numServices)
}

func (fd *File) allocEnums(n int) []Enum {
	total := len(fd.allEnums)
	es := fd.allEnums[total : total+n]
	fd.allEnums = fd.allEnums[:total+n]
	return es
}
func (fd *File) allocMessages(n int) []Message {
	total := len(fd.allMessages)
	ms := fd.allMessages[total : total+n]
	fd.allMessages = fd.allMessages[:total+n]
	return ms
}
func (fd *File) allocExtensions(n int) []Extension {
	total := len(fd.allExtensions)
	xs := fd.allExtensions[total : total+n]
	fd.allExtensions = fd.allExtensions[:total+n]
	return xs
}
func (fd *File) allocServices(n int) []Service {
	total := len(fd.allServices)
	xs := fd.allServices[total : total+n]
	fd.allServices = fd.allServices[:total+n]
	return xs
}

// checkDecls performs a sanity check that the expected number of expected
// declarations matches the number that were found in the descriptor proto.
func (fd *File) checkDecls() {
	switch {
	case len(fd.allEnums) != cap(fd.allEnums):
	case len(fd.allMessages) != cap(fd.allMessages):
	case len(fd.allExtensions) != cap(fd.allExtensions):
	case len(fd.allServices) != cap(fd.allServices):
	default:
		return
	}
	panic("mismatching cardinality")
}

func (fd *File) unmarshalSeed(b []byte) {
	sb := getBuilder()
	defer putBuilder(sb)

	var prevField protoreflect.FieldNumber
	var numEnums, numMessages, numExtensions, numServices int
	var posEnums, posMessages, posExtensions, posServices int
	var options []byte
	b0 := b
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FileDescriptorProto_Syntax_field_number:
				switch string(v) {
				case "proto2":
					fd.L1.Syntax = protoreflect.Proto2
				case "proto3":
					fd.L1.Syntax = protoreflect.Proto3
				case "editions":
					fd.L1.Syntax = protoreflect.Editions
				default:
					panic("invalid syntax")
				}
			case genid.FileDescriptorProto_Name_field_number:
				fd.L1.Path = sb.MakeString(v)
			case genid.FileDescriptorProto_Package_field_number:
				fd.L1.Package = protoreflect.FullName(sb.MakeString(v))
			case genid.FileDescriptorProto_Options_field_number:
				options = v
			case genid.FileDescriptorProto_EnumType_field_number:
				if prevField != genid.FileDescriptorProto_EnumType_field_number {
					if numEnums > 0 {
						panic("non-contiguous repeated field")
					}
					posEnums = len(b0) - len(b) - n - m
				}
				numEnums++
			case genid.FileDescriptorProto_MessageType_field_number:
				if prevField != genid.FileDescriptorProto_MessageType_field_number {
					if numMessages > 0 {
						panic("non-contiguous repeated field")
					}
					posMessages = len(b0) - len(b) - n - m
				}
				numMessages++
			case genid.FileDescriptorProto_Extension_field_number:
				if prevField != genid.FileDescriptorProto_Extension_field_number {
					if numExtensions > 0 {
						panic("non-contiguous repeated field")
					}
					posExtensions = len(b0) - len(b) - n - m
				}
				numExtensions++
			case genid.FileDescriptorProto_Service_field_number:
				if prevField != genid.FileDescriptorProto_Service_field_number {
					if numServices > 0 {
						panic("non-contiguous repeated field")
					}
					posServices = len(b0) - len(b) - n - m
				}
				numServices++
			}
			prevField = num
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FileDescriptorProto_Edition_field_number:
				fd.L1.Edition = Edition(v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
			prevField = -1 // ignore known field numbers of unknown wire type
		}
	}

	// If syntax is missing, it is assumed to be proto2.
	if fd.L1.Syntax == 0 {
		fd.L1.Syntax = protoreflect.Proto2
	}

	if fd.L1.Syntax == protoreflect.Editions {
		fd.L1.EditionFeatures = getFeaturesFor(fd.L1.Edition)
	}

	// Parse editions features from options if any
	if options != nil {
		fd.unmarshalSeedOptions(options)
	}

	// Must allocate all declarations before parsing each descriptor type
	// to ensure we handled all descriptors in "flattened ordering".
	if numEnums > 0 {
		fd.L1.Enums.List = fd.allocEnums(numEnums)
	}
	if numMessages > 0 {
		fd.L1.Messages.List = fd.allocMessages(numMessages)
	}
	if numExtensions > 0 {
		fd.L1.Extensions.List = fd.allocExtensions(numExtensions)
	}
	if numServices > 0 {
		fd.L1.Services.List = fd.allocServices(numServices)
	}

	if numEnums > 0 {
		b := b0[posEnums:]
		for i := range fd.L1.Enums.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			fd.L1.Enums.List[i].unmarshalSeed(v, sb, fd, fd, i)
			b = b[n+m:]
		}
	}
	if numMessages > 0 {
		b := b0[posMessages:]
		for i := range fd.L1.Messages.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			fd.L1.Messages.List[i].unmarshalSeed(v, sb, fd, fd, i)
			b = b[n+m:]
		}
	}
	if numExtensions > 0 {
		b := b0[posExtensions:]
		for i := range fd.L1.Extensions.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			fd.L1.Extensions.List[i].unmarshalSeed(v, sb, fd, fd, i)
			b = b[n+m:]
		}
	}
	if numServices > 0 {
		b := b0[posServices:]
		for i := range fd.L1.Services.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			fd.L1.Services.List[i].unmarshalSeed(v, sb, fd, fd, i)
			b = b[n+m:]
		}
	}
}

func (fd *File) unmarshalSeedOptions(b []byte) {
	for b := b; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FileOptions_Features_field_number:
				if fd.Syntax() != protoreflect.Editions {
					panic(fmt.Sprintf("invalid descriptor: using edition features in a proto with syntax %s", fd.Syntax()))
				}
				fd.L1.EditionFeatures = unmarshalFeatureSet(v, fd.L1.EditionFeatures)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (ed *Enum) unmarshalSeed(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	ed.L0.ParentFile = pf
	ed.L0.Parent = pd
	ed.L0.Index = i

	var numValues int
	for b := b; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.EnumDescriptorProto_Name_field_number:
				ed.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.EnumDescriptorProto_Value_field_number:
				numValues++
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}

	// Only construct enum value descriptors for top-level enums since
	// they are needed for registration.
	if pd != pf {
		return
	}
	ed.L1.eagerValues = true
	ed.L2 = new(EnumL2)
	ed.L2.Values.List = make([]EnumValue, numValues)
	for i := 0; len(b) > 0; {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.EnumDescriptorProto_Value_field_number:
				ed.L2.Values.List[i].unmarshalFull(v, sb, pf, ed, i)
				i++
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (md *Message) unmarshalSeed(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	md.L0.ParentFile = pf
	md.L0.Parent = pd
	md.L0.Index = i

	var prevField protoreflect.FieldNumber
	var numEnums, numMessages, numExtensions int
	var posEnums, posMessages, posExtensions int
	b0 := b
	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.DescriptorProto_Name_field_number:
				md.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.DescriptorProto_EnumType_field_number:
				if prevField != genid.DescriptorProto_EnumType_field_number {
					if numEnums > 0 {
						panic("non-contiguous repeated field")
					}
					posEnums = len(b0) - len(b) - n - m
				}
				numEnums++
			case genid.DescriptorProto_NestedType_field_number:
				if prevField != genid.DescriptorProto_NestedType_field_number {
					if numMessages > 0 {
						panic("non-contiguous repeated field")
					}
					posMessages = len(b0) - len(b) - n - m
				}
				numMessages++
			case genid.DescriptorProto_Extension_field_number:
				if prevField != genid.DescriptorProto_Extension_field_number {
					if numExtensions > 0 {
						panic("non-contiguous repeated field")
					}
					posExtensions = len(b0) - len(b) - n - m
				}
				numExtensions++
			case genid.DescriptorProto_Options_field_number:
				md.unmarshalSeedOptions(v)
			}
			prevField = num
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
			prevField = -1 // ignore known field numbers of unknown wire type
		}
	}

	// Must allocate all declarations before parsing each descriptor type
	// to ensure we handled all descriptors in "flattened ordering".
	if numEnums > 0 {
		md.L1.Enums.List = pf.allocEnums(numEnums)
	}
	if numMessages > 0 {
		md.L1.Messages.List = pf.allocMessages(numMessages)
	}
	if numExtensions > 0 {
		md.L1.Extensions.List = pf.allocExtensions(numExtensions)
	}

	if numEnums > 0 {
		b := b0[posEnums:]
		for i := range md.L1.Enums.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			md.L1.Enums.List[i].unmarshalSeed(v, sb, pf, md, i)
			b = b[n+m:]
		}
	}
	if numMessages > 0 {
		b := b0[posMessages:]
		for i := range md.L1.Messages.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			md.L1.Messages.List[i].unmarshalSeed(v, sb, pf, md, i)
			b = b[n+m:]
		}
	}
	if numExtensions > 0 {
		b := b0[posExtensions:]
		for i := range md.L1.Extensions.List {
			_, n := protowire.ConsumeVarint(b)
			v, m := protowire.ConsumeBytes(b[n:])
			md.L1.Extensions.List[i].unmarshalSeed(v, sb, pf, md, i)
			b = b[n+m:]
		}
	}
}

func (md *Message) unmarshalSeedOptions(b []byte) {
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
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (xd *Extension) unmarshalSeed(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	xd.L0.ParentFile = pf
	xd.L0.Parent = pd
	xd.L0.Index = i

	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.VarintType:
			v, m := protowire.ConsumeVarint(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_Number_field_number:
				xd.L1.Number = protoreflect.FieldNumber(v)
			case genid.FieldDescriptorProto_Label_field_number:
				xd.L1.Cardinality = protoreflect.Cardinality(v)
			case genid.FieldDescriptorProto_Type_field_number:
				xd.L1.Kind = protoreflect.Kind(v)
			}
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.FieldDescriptorProto_Name_field_number:
				xd.L0.FullName = appendFullName(sb, pd.FullName(), v)
			case genid.FieldDescriptorProto_Extendee_field_number:
				xd.L1.Extendee = PlaceholderMessage(makeFullName(sb, v))
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

func (sd *Service) unmarshalSeed(b []byte, sb *strs.Builder, pf *File, pd protoreflect.Descriptor, i int) {
	sd.L0.ParentFile = pf
	sd.L0.Parent = pd
	sd.L0.Index = i

	for len(b) > 0 {
		num, typ, n := protowire.ConsumeTag(b)
		b = b[n:]
		switch typ {
		case protowire.BytesType:
			v, m := protowire.ConsumeBytes(b)
			b = b[m:]
			switch num {
			case genid.ServiceDescriptorProto_Name_field_number:
				sd.L0.FullName = appendFullName(sb, pd.FullName(), v)
			}
		default:
			m := protowire.ConsumeFieldValue(num, typ, b)
			b = b[m:]
		}
	}
}

var nameBuilderPool = sync.Pool{
	New: func() interface{} { return new(strs.Builder) },
}

func getBuilder() *strs.Builder {
	return nameBuilderPool.Get().(*strs.Builder)
}
func putBuilder(b *strs.Builder) {
	nameBuilderPool.Put(b)
}

// makeFullName converts b to a protoreflect.FullName,
// where b must start with a leading dot.
func makeFullName(sb *strs.Builder, b []byte) protoreflect.FullName {
	if len(b) == 0 || b[0] != '.' {
		panic("name reference must be fully qualified")
	}
	return protoreflect.FullName(sb.MakeString(b[1:]))
}

func appendFullName(sb *strs.Builder, prefix protoreflect.FullName, suffix []byte) protoreflect.FullName {
	return sb.AppendFullName(prefix, protoreflect.Name(strs.UnsafeString(suffix)))
}
