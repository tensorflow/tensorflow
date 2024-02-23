// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal_gengo

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"

	"google.golang.org/protobuf/compiler/protogen"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protopath"
	"google.golang.org/protobuf/reflect/protorange"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

func genReflectFileDescriptor(gen *protogen.Plugin, g *protogen.GeneratedFile, f *fileInfo) {
	g.P("var ", f.GoDescriptorIdent, " ", protoreflectPackage.Ident("FileDescriptor"))
	g.P()

	genFileDescriptor(gen, g, f)
	if len(f.allEnums) > 0 {
		g.P("var ", enumTypesVarName(f), " = make([]", protoimplPackage.Ident("EnumInfo"), ",", len(f.allEnums), ")")
	}
	if len(f.allMessages) > 0 {
		g.P("var ", messageTypesVarName(f), " = make([]", protoimplPackage.Ident("MessageInfo"), ",", len(f.allMessages), ")")
	}

	// Generate a unique list of Go types for all declarations and dependencies,
	// and the associated index into the type list for all dependencies.
	var goTypes []string
	var depIdxs []string
	seen := map[protoreflect.FullName]int{}
	genDep := func(name protoreflect.FullName, depSource string) {
		if depSource != "" {
			line := fmt.Sprintf("%d, // %d: %s -> %s", seen[name], len(depIdxs), depSource, name)
			depIdxs = append(depIdxs, line)
		}
	}
	genEnum := func(e *protogen.Enum, depSource string) {
		if e != nil {
			name := e.Desc.FullName()
			if _, ok := seen[name]; !ok {
				line := fmt.Sprintf("(%s)(0), // %d: %s", g.QualifiedGoIdent(e.GoIdent), len(goTypes), name)
				goTypes = append(goTypes, line)
				seen[name] = len(seen)
			}
			if depSource != "" {
				genDep(name, depSource)
			}
		}
	}
	genMessage := func(m *protogen.Message, depSource string) {
		if m != nil {
			name := m.Desc.FullName()
			if _, ok := seen[name]; !ok {
				line := fmt.Sprintf("(*%s)(nil), // %d: %s", g.QualifiedGoIdent(m.GoIdent), len(goTypes), name)
				if m.Desc.IsMapEntry() {
					// Map entry messages have no associated Go type.
					line = fmt.Sprintf("nil, // %d: %s", len(goTypes), name)
				}
				goTypes = append(goTypes, line)
				seen[name] = len(seen)
			}
			if depSource != "" {
				genDep(name, depSource)
			}
		}
	}

	// This ordering is significant.
	// See filetype.TypeBuilder.DependencyIndexes.
	type offsetEntry struct {
		start int
		name  string
	}
	var depOffsets []offsetEntry
	for _, enum := range f.allEnums {
		genEnum(enum.Enum, "")
	}
	for _, message := range f.allMessages {
		genMessage(message.Message, "")
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), "field type_name"})
	for _, message := range f.allMessages {
		for _, field := range message.Fields {
			if field.Desc.IsWeak() {
				continue
			}
			source := string(field.Desc.FullName())
			genEnum(field.Enum, source+":type_name")
			genMessage(field.Message, source+":type_name")
		}
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), "extension extendee"})
	for _, extension := range f.allExtensions {
		source := string(extension.Desc.FullName())
		genMessage(extension.Extendee, source+":extendee")
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), "extension type_name"})
	for _, extension := range f.allExtensions {
		source := string(extension.Desc.FullName())
		genEnum(extension.Enum, source+":type_name")
		genMessage(extension.Message, source+":type_name")
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), "method input_type"})
	for _, service := range f.Services {
		for _, method := range service.Methods {
			source := string(method.Desc.FullName())
			genMessage(method.Input, source+":input_type")
		}
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), "method output_type"})
	for _, service := range f.Services {
		for _, method := range service.Methods {
			source := string(method.Desc.FullName())
			genMessage(method.Output, source+":output_type")
		}
	}
	depOffsets = append(depOffsets, offsetEntry{len(depIdxs), ""})
	for i := len(depOffsets) - 2; i >= 0; i-- {
		curr, next := depOffsets[i], depOffsets[i+1]
		depIdxs = append(depIdxs, fmt.Sprintf("%d, // [%d:%d] is the sub-list for %s",
			curr.start, curr.start, next.start, curr.name))
	}
	if len(depIdxs) > math.MaxInt32 {
		panic("too many dependencies") // sanity check
	}

	g.P("var ", goTypesVarName(f), " = []interface{}{")
	for _, s := range goTypes {
		g.P(s)
	}
	g.P("}")

	g.P("var ", depIdxsVarName(f), " = []int32{")
	for _, s := range depIdxs {
		g.P(s)
	}
	g.P("}")

	g.P("func init() { ", initFuncName(f.File), "() }")

	g.P("func ", initFuncName(f.File), "() {")
	g.P("if ", f.GoDescriptorIdent, " != nil {")
	g.P("return")
	g.P("}")

	// Ensure that initialization functions for different files in the same Go
	// package run in the correct order: Call the init funcs for every .proto file
	// imported by this one that is in the same Go package.
	for i, imps := 0, f.Desc.Imports(); i < imps.Len(); i++ {
		impFile := gen.FilesByPath[imps.Get(i).Path()]
		if impFile.GoImportPath != f.GoImportPath {
			continue
		}
		g.P(initFuncName(impFile), "()")
	}

	if len(f.allMessages) > 0 {
		// Populate MessageInfo.Exporters.
		g.P("if !", protoimplPackage.Ident("UnsafeEnabled"), " {")
		for _, message := range f.allMessages {
			if sf := f.allMessageFieldsByPtr[message]; len(sf.unexported) > 0 {
				idx := f.allMessagesByPtr[message]
				typesVar := messageTypesVarName(f)

				g.P(typesVar, "[", idx, "].Exporter = func(v interface{}, i int) interface{} {")
				g.P("switch v := v.(*", message.GoIdent, "); i {")
				for i := 0; i < sf.count; i++ {
					if name := sf.unexported[i]; name != "" {
						g.P("case ", i, ": return &v.", name)
					}
				}
				g.P("default: return nil")
				g.P("}")
				g.P("}")
			}
		}
		g.P("}")

		// Populate MessageInfo.OneofWrappers.
		for _, message := range f.allMessages {
			if len(message.Oneofs) > 0 {
				idx := f.allMessagesByPtr[message]
				typesVar := messageTypesVarName(f)

				// Associate the wrapper types by directly passing them to the MessageInfo.
				g.P(typesVar, "[", idx, "].OneofWrappers = []interface{} {")
				for _, oneof := range message.Oneofs {
					if !oneof.Desc.IsSynthetic() {
						for _, field := range oneof.Fields {
							g.P("(*", field.GoIdent, ")(nil),")
						}
					}
				}
				g.P("}")
			}
		}
	}

	g.P("type x struct{}")
	g.P("out := ", protoimplPackage.Ident("TypeBuilder"), "{")
	g.P("File: ", protoimplPackage.Ident("DescBuilder"), "{")
	g.P("GoPackagePath: ", reflectPackage.Ident("TypeOf"), "(x{}).PkgPath(),")
	g.P("RawDescriptor: ", rawDescVarName(f), ",")
	g.P("NumEnums: ", len(f.allEnums), ",")
	g.P("NumMessages: ", len(f.allMessages), ",")
	g.P("NumExtensions: ", len(f.allExtensions), ",")
	g.P("NumServices: ", len(f.Services), ",")
	g.P("},")
	g.P("GoTypes: ", goTypesVarName(f), ",")
	g.P("DependencyIndexes: ", depIdxsVarName(f), ",")
	if len(f.allEnums) > 0 {
		g.P("EnumInfos: ", enumTypesVarName(f), ",")
	}
	if len(f.allMessages) > 0 {
		g.P("MessageInfos: ", messageTypesVarName(f), ",")
	}
	if len(f.allExtensions) > 0 {
		g.P("ExtensionInfos: ", extensionTypesVarName(f), ",")
	}
	g.P("}.Build()")
	g.P(f.GoDescriptorIdent, " = out.File")

	// Set inputs to nil to allow GC to reclaim resources.
	g.P(rawDescVarName(f), " = nil")
	g.P(goTypesVarName(f), " = nil")
	g.P(depIdxsVarName(f), " = nil")
	g.P("}")
}

// stripSourceRetentionFieldsFromMessage walks the given message tree recursively
// and clears any fields with the field option: [retention = RETENTION_SOURCE]
func stripSourceRetentionFieldsFromMessage(m protoreflect.Message) {
	protorange.Range(m, func(ppv protopath.Values) error {
		m2, ok := ppv.Index(-1).Value.Interface().(protoreflect.Message)
		if !ok {
			return nil
		}
		m2.Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
			fdo, ok := fd.Options().(*descriptorpb.FieldOptions)
			if ok && fdo.GetRetention() == descriptorpb.FieldOptions_RETENTION_SOURCE {
				m2.Clear(fd)
			}
			return true
		})
		return nil
	})
}

func genFileDescriptor(gen *protogen.Plugin, g *protogen.GeneratedFile, f *fileInfo) {
	descProto := proto.Clone(f.Proto).(*descriptorpb.FileDescriptorProto)
	descProto.SourceCodeInfo = nil // drop source code information
	stripSourceRetentionFieldsFromMessage(descProto.ProtoReflect())
	b, err := proto.MarshalOptions{AllowPartial: true, Deterministic: true}.Marshal(descProto)
	if err != nil {
		gen.Error(err)
		return
	}

	g.P("var ", rawDescVarName(f), " = []byte{")
	for len(b) > 0 {
		n := 16
		if n > len(b) {
			n = len(b)
		}

		s := ""
		for _, c := range b[:n] {
			s += fmt.Sprintf("0x%02x,", c)
		}
		g.P(s)

		b = b[n:]
	}
	g.P("}")
	g.P()

	if f.needRawDesc {
		onceVar := rawDescVarName(f) + "Once"
		dataVar := rawDescVarName(f) + "Data"
		g.P("var (")
		g.P(onceVar, " ", syncPackage.Ident("Once"))
		g.P(dataVar, " = ", rawDescVarName(f))
		g.P(")")
		g.P()

		g.P("func ", rawDescVarName(f), "GZIP() []byte {")
		g.P(onceVar, ".Do(func() {")
		g.P(dataVar, " = ", protoimplPackage.Ident("X"), ".CompressGZIP(", dataVar, ")")
		g.P("})")
		g.P("return ", dataVar)
		g.P("}")
		g.P()
	}
}

func genEnumReflectMethods(g *protogen.GeneratedFile, f *fileInfo, e *enumInfo) {
	idx := f.allEnumsByPtr[e]
	typesVar := enumTypesVarName(f)

	// Descriptor method.
	g.P("func (", e.GoIdent, ") Descriptor() ", protoreflectPackage.Ident("EnumDescriptor"), " {")
	g.P("return ", typesVar, "[", idx, "].Descriptor()")
	g.P("}")
	g.P()

	// Type method.
	g.P("func (", e.GoIdent, ") Type() ", protoreflectPackage.Ident("EnumType"), " {")
	g.P("return &", typesVar, "[", idx, "]")
	g.P("}")
	g.P()

	// Number method.
	g.P("func (x ", e.GoIdent, ") Number() ", protoreflectPackage.Ident("EnumNumber"), " {")
	g.P("return ", protoreflectPackage.Ident("EnumNumber"), "(x)")
	g.P("}")
	g.P()
}

func genMessageReflectMethods(g *protogen.GeneratedFile, f *fileInfo, m *messageInfo) {
	idx := f.allMessagesByPtr[m]
	typesVar := messageTypesVarName(f)

	// ProtoReflect method.
	g.P("func (x *", m.GoIdent, ") ProtoReflect() ", protoreflectPackage.Ident("Message"), " {")
	g.P("mi := &", typesVar, "[", idx, "]")
	g.P("if ", protoimplPackage.Ident("UnsafeEnabled"), " && x != nil {")
	g.P("ms := ", protoimplPackage.Ident("X"), ".MessageStateOf(", protoimplPackage.Ident("Pointer"), "(x))")
	g.P("if ms.LoadMessageInfo() == nil {")
	g.P("ms.StoreMessageInfo(mi)")
	g.P("}")
	g.P("return ms")
	g.P("}")
	g.P("return mi.MessageOf(x)")
	g.P("}")
	g.P()
}

func fileVarName(f *protogen.File, suffix string) string {
	prefix := f.GoDescriptorIdent.GoName
	_, n := utf8.DecodeRuneInString(prefix)
	prefix = strings.ToLower(prefix[:n]) + prefix[n:]
	return prefix + "_" + suffix
}
func rawDescVarName(f *fileInfo) string {
	return fileVarName(f.File, "rawDesc")
}
func goTypesVarName(f *fileInfo) string {
	return fileVarName(f.File, "goTypes")
}
func depIdxsVarName(f *fileInfo) string {
	return fileVarName(f.File, "depIdxs")
}
func enumTypesVarName(f *fileInfo) string {
	return fileVarName(f.File, "enumTypes")
}
func messageTypesVarName(f *fileInfo) string {
	return fileVarName(f.File, "msgTypes")
}
func extensionTypesVarName(f *fileInfo) string {
	return fileVarName(f.File, "extTypes")
}
func initFuncName(f *protogen.File) string {
	return fileVarName(f, "init")
}
