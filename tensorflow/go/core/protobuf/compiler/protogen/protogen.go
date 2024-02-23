// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protogen provides support for writing protoc plugins.
//
// Plugins for protoc, the Protocol Buffer compiler,
// are programs which read a [pluginpb.CodeGeneratorRequest] message from standard input
// and write a [pluginpb.CodeGeneratorResponse] message to standard output.
// This package provides support for writing plugins which generate Go code.
package protogen

import (
	"bufio"
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"
	"google.golang.org/protobuf/types/pluginpb"
)

const goPackageDocURL = "https://protobuf.dev/reference/go/go-generated#package"

// Run executes a function as a protoc plugin.
//
// It reads a [pluginpb.CodeGeneratorRequest] message from [os.Stdin], invokes the plugin
// function, and writes a [pluginpb.CodeGeneratorResponse] message to [os.Stdout].
//
// If a failure occurs while reading or writing, Run prints an error to
// [os.Stderr] and calls [os.Exit](1).
func (opts Options) Run(f func(*Plugin) error) {
	if err := run(opts, f); err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", filepath.Base(os.Args[0]), err)
		os.Exit(1)
	}
}

func run(opts Options, f func(*Plugin) error) error {
	if len(os.Args) > 1 {
		return fmt.Errorf("unknown argument %q (this program should be run by protoc, not directly)", os.Args[1])
	}
	in, err := ioutil.ReadAll(os.Stdin)
	if err != nil {
		return err
	}
	req := &pluginpb.CodeGeneratorRequest{}
	if err := proto.Unmarshal(in, req); err != nil {
		return err
	}
	gen, err := opts.New(req)
	if err != nil {
		return err
	}
	if err := f(gen); err != nil {
		// Errors from the plugin function are reported by setting the
		// error field in the CodeGeneratorResponse.
		//
		// In contrast, errors that indicate a problem in protoc
		// itself (unparsable input, I/O errors, etc.) are reported
		// to stderr.
		gen.Error(err)
	}
	resp := gen.Response()
	out, err := proto.Marshal(resp)
	if err != nil {
		return err
	}
	if _, err := os.Stdout.Write(out); err != nil {
		return err
	}
	return nil
}

// A Plugin is a protoc plugin invocation.
type Plugin struct {
	// Request is the CodeGeneratorRequest provided by protoc.
	Request *pluginpb.CodeGeneratorRequest

	// Files is the set of files to generate and everything they import.
	// Files appear in topological order, so each file appears before any
	// file that imports it.
	Files       []*File
	FilesByPath map[string]*File

	// SupportedFeatures is the set of protobuf language features supported by
	// this generator plugin. See the documentation for
	// google.protobuf.CodeGeneratorResponse.supported_features for details.
	SupportedFeatures uint64

	fileReg        *protoregistry.Files
	enumsByName    map[protoreflect.FullName]*Enum
	messagesByName map[protoreflect.FullName]*Message
	annotateCode   bool
	pathType       pathType
	module         string
	genFiles       []*GeneratedFile
	opts           Options
	err            error
}

type Options struct {
	// If ParamFunc is non-nil, it will be called with each unknown
	// generator parameter.
	//
	// Plugins for protoc can accept parameters from the command line,
	// passed in the --<lang>_out protoc, separated from the output
	// directory with a colon; e.g.,
	//
	//   --go_out=<param1>=<value1>,<param2>=<value2>:<output_directory>
	//
	// Parameters passed in this fashion as a comma-separated list of
	// key=value pairs will be passed to the ParamFunc.
	//
	// The (flag.FlagSet).Set method matches this function signature,
	// so parameters can be converted into flags as in the following:
	//
	//   var flags flag.FlagSet
	//   value := flags.Bool("param", false, "")
	//   opts := &protogen.Options{
	//     ParamFunc: flags.Set,
	//   }
	//   protogen.Run(opts, func(p *protogen.Plugin) error {
	//     if *value { ... }
	//   })
	ParamFunc func(name, value string) error

	// ImportRewriteFunc is called with the import path of each package
	// imported by a generated file. It returns the import path to use
	// for this package.
	ImportRewriteFunc func(GoImportPath) GoImportPath
}

// New returns a new Plugin.
func (opts Options) New(req *pluginpb.CodeGeneratorRequest) (*Plugin, error) {
	gen := &Plugin{
		Request:        req,
		FilesByPath:    make(map[string]*File),
		fileReg:        new(protoregistry.Files),
		enumsByName:    make(map[protoreflect.FullName]*Enum),
		messagesByName: make(map[protoreflect.FullName]*Message),
		opts:           opts,
	}

	packageNames := make(map[string]GoPackageName) // filename -> package name
	importPaths := make(map[string]GoImportPath)   // filename -> import path
	for _, param := range strings.Split(req.GetParameter(), ",") {
		var value string
		if i := strings.Index(param, "="); i >= 0 {
			value = param[i+1:]
			param = param[0:i]
		}
		switch param {
		case "":
			// Ignore.
		case "module":
			gen.module = value
		case "paths":
			switch value {
			case "import":
				gen.pathType = pathTypeImport
			case "source_relative":
				gen.pathType = pathTypeSourceRelative
			default:
				return nil, fmt.Errorf(`unknown path type %q: want "import" or "source_relative"`, value)
			}
		case "annotate_code":
			switch value {
			case "true", "":
				gen.annotateCode = true
			case "false":
			default:
				return nil, fmt.Errorf(`bad value for parameter %q: want "true" or "false"`, param)
			}
		default:
			if param[0] == 'M' {
				impPath, pkgName := splitImportPathAndPackageName(value)
				if pkgName != "" {
					packageNames[param[1:]] = pkgName
				}
				if impPath != "" {
					importPaths[param[1:]] = impPath
				}
				continue
			}
			if opts.ParamFunc != nil {
				if err := opts.ParamFunc(param, value); err != nil {
					return nil, err
				}
			}
		}
	}

	// When the module= option is provided, we strip the module name
	// prefix from generated files. This only makes sense if generated
	// filenames are based on the import path.
	if gen.module != "" && gen.pathType == pathTypeSourceRelative {
		return nil, fmt.Errorf("cannot use module= with paths=source_relative")
	}

	// Figure out the import path and package name for each file.
	//
	// The rules here are complicated and have grown organically over time.
	// Interactions between different ways of specifying package information
	// may be surprising.
	//
	// The recommended approach is to include a go_package option in every
	// .proto source file specifying the full import path of the Go package
	// associated with this file.
	//
	//     option go_package = "google.golang.org/protobuf/types/known/anypb";
	//
	// Alternatively, build systems which want to exert full control over
	// import paths may specify M<filename>=<import_path> flags.
	for _, fdesc := range gen.Request.ProtoFile {
		// The "M" command-line flags take precedence over
		// the "go_package" option in the .proto source file.
		filename := fdesc.GetName()
		impPath, pkgName := splitImportPathAndPackageName(fdesc.GetOptions().GetGoPackage())
		if importPaths[filename] == "" && impPath != "" {
			importPaths[filename] = impPath
		}
		if packageNames[filename] == "" && pkgName != "" {
			packageNames[filename] = pkgName
		}
		switch {
		case importPaths[filename] == "":
			// The import path must be specified one way or another.
			return nil, fmt.Errorf(
				"unable to determine Go import path for %q\n\n"+
					"Please specify either:\n"+
					"\t• a \"go_package\" option in the .proto source file, or\n"+
					"\t• a \"M\" argument on the command line.\n\n"+
					"See %v for more information.\n",
				fdesc.GetName(), goPackageDocURL)
		case !strings.Contains(string(importPaths[filename]), ".") &&
			!strings.Contains(string(importPaths[filename]), "/"):
			// Check that import paths contain at least a dot or slash to avoid
			// a common mistake where import path is confused with package name.
			return nil, fmt.Errorf(
				"invalid Go import path %q for %q\n\n"+
					"The import path must contain at least one period ('.') or forward slash ('/') character.\n\n"+
					"See %v for more information.\n",
				string(importPaths[filename]), fdesc.GetName(), goPackageDocURL)
		case packageNames[filename] == "":
			// If the package name is not explicitly specified,
			// then derive a reasonable package name from the import path.
			//
			// NOTE: The package name is derived first from the import path in
			// the "go_package" option (if present) before trying the "M" flag.
			// The inverted order for this is because the primary use of the "M"
			// flag is by build systems that have full control over the
			// import paths all packages, where it is generally expected that
			// the Go package name still be identical for the Go toolchain and
			// for custom build systems like Bazel.
			if impPath == "" {
				impPath = importPaths[filename]
			}
			packageNames[filename] = cleanPackageName(path.Base(string(impPath)))
		}
	}

	// Consistency check: Every file with the same Go import path should have
	// the same Go package name.
	packageFiles := make(map[GoImportPath][]string)
	for filename, importPath := range importPaths {
		if _, ok := packageNames[filename]; !ok {
			// Skip files mentioned in a M<file>=<import_path> parameter
			// but which do not appear in the CodeGeneratorRequest.
			continue
		}
		packageFiles[importPath] = append(packageFiles[importPath], filename)
	}
	for importPath, filenames := range packageFiles {
		for i := 1; i < len(filenames); i++ {
			if a, b := packageNames[filenames[0]], packageNames[filenames[i]]; a != b {
				return nil, fmt.Errorf("Go package %v has inconsistent names %v (%v) and %v (%v)",
					importPath, a, filenames[0], b, filenames[i])
			}
		}
	}

	// The extracted types from the full import set
	typeRegistry := newExtensionRegistry()
	for _, fdesc := range gen.Request.ProtoFile {
		filename := fdesc.GetName()
		if gen.FilesByPath[filename] != nil {
			return nil, fmt.Errorf("duplicate file name: %q", filename)
		}
		f, err := newFile(gen, fdesc, packageNames[filename], importPaths[filename])
		if err != nil {
			return nil, err
		}
		gen.Files = append(gen.Files, f)
		gen.FilesByPath[filename] = f
		if err = typeRegistry.registerAllExtensionsFromFile(f.Desc); err != nil {
			return nil, err
		}
	}
	for _, filename := range gen.Request.FileToGenerate {
		f, ok := gen.FilesByPath[filename]
		if !ok {
			return nil, fmt.Errorf("no descriptor for generated file: %v", filename)
		}
		f.Generate = true
	}

	// Create fully-linked descriptors if new extensions were found
	if typeRegistry.hasNovelExtensions() {
		for _, f := range gen.Files {
			b, err := proto.Marshal(f.Proto.ProtoReflect().Interface())
			if err != nil {
				return nil, err
			}
			err = proto.UnmarshalOptions{Resolver: typeRegistry}.Unmarshal(b, f.Proto)
			if err != nil {
				return nil, err
			}
		}
	}
	return gen, nil
}

// Error records an error in code generation. The generator will report the
// error back to protoc and will not produce output.
func (gen *Plugin) Error(err error) {
	if gen.err == nil {
		gen.err = err
	}
}

// Response returns the generator output.
func (gen *Plugin) Response() *pluginpb.CodeGeneratorResponse {
	resp := &pluginpb.CodeGeneratorResponse{}
	if gen.err != nil {
		resp.Error = proto.String(gen.err.Error())
		return resp
	}
	for _, g := range gen.genFiles {
		if g.skip {
			continue
		}
		content, err := g.Content()
		if err != nil {
			return &pluginpb.CodeGeneratorResponse{
				Error: proto.String(err.Error()),
			}
		}
		filename := g.filename
		if gen.module != "" {
			trim := gen.module + "/"
			if !strings.HasPrefix(filename, trim) {
				return &pluginpb.CodeGeneratorResponse{
					Error: proto.String(fmt.Sprintf("%v: generated file does not match prefix %q", filename, gen.module)),
				}
			}
			filename = strings.TrimPrefix(filename, trim)
		}
		resp.File = append(resp.File, &pluginpb.CodeGeneratorResponse_File{
			Name:    proto.String(filename),
			Content: proto.String(string(content)),
		})
		if gen.annotateCode && strings.HasSuffix(g.filename, ".go") {
			meta, err := g.metaFile(content)
			if err != nil {
				return &pluginpb.CodeGeneratorResponse{
					Error: proto.String(err.Error()),
				}
			}
			resp.File = append(resp.File, &pluginpb.CodeGeneratorResponse_File{
				Name:    proto.String(filename + ".meta"),
				Content: proto.String(meta),
			})
		}
	}
	if gen.SupportedFeatures > 0 {
		resp.SupportedFeatures = proto.Uint64(gen.SupportedFeatures)
	}
	return resp
}

// A File describes a .proto source file.
type File struct {
	Desc  protoreflect.FileDescriptor
	Proto *descriptorpb.FileDescriptorProto

	GoDescriptorIdent GoIdent       // name of Go variable for the file descriptor
	GoPackageName     GoPackageName // name of this file's Go package
	GoImportPath      GoImportPath  // import path of this file's Go package

	Enums      []*Enum      // top-level enum declarations
	Messages   []*Message   // top-level message declarations
	Extensions []*Extension // top-level extension declarations
	Services   []*Service   // top-level service declarations

	Generate bool // true if we should generate code for this file

	// GeneratedFilenamePrefix is used to construct filenames for generated
	// files associated with this source file.
	//
	// For example, the source file "dir/foo.proto" might have a filename prefix
	// of "dir/foo". Appending ".pb.go" produces an output file of "dir/foo.pb.go".
	GeneratedFilenamePrefix string

	location Location
}

func newFile(gen *Plugin, p *descriptorpb.FileDescriptorProto, packageName GoPackageName, importPath GoImportPath) (*File, error) {
	desc, err := protodesc.NewFile(p, gen.fileReg)
	if err != nil {
		return nil, fmt.Errorf("invalid FileDescriptorProto %q: %v", p.GetName(), err)
	}
	if err := gen.fileReg.RegisterFile(desc); err != nil {
		return nil, fmt.Errorf("cannot register descriptor %q: %v", p.GetName(), err)
	}
	f := &File{
		Desc:          desc,
		Proto:         p,
		GoPackageName: packageName,
		GoImportPath:  importPath,
		location:      Location{SourceFile: desc.Path()},
	}

	// Determine the prefix for generated Go files.
	prefix := p.GetName()
	if ext := path.Ext(prefix); ext == ".proto" || ext == ".protodevel" {
		prefix = prefix[:len(prefix)-len(ext)]
	}
	switch gen.pathType {
	case pathTypeImport:
		// If paths=import, the output filename is derived from the Go import path.
		prefix = path.Join(string(f.GoImportPath), path.Base(prefix))
	case pathTypeSourceRelative:
		// If paths=source_relative, the output filename is derived from
		// the input filename.
	}
	f.GoDescriptorIdent = GoIdent{
		GoName:       "File_" + strs.GoSanitized(p.GetName()),
		GoImportPath: f.GoImportPath,
	}
	f.GeneratedFilenamePrefix = prefix

	for i, eds := 0, desc.Enums(); i < eds.Len(); i++ {
		f.Enums = append(f.Enums, newEnum(gen, f, nil, eds.Get(i)))
	}
	for i, mds := 0, desc.Messages(); i < mds.Len(); i++ {
		f.Messages = append(f.Messages, newMessage(gen, f, nil, mds.Get(i)))
	}
	for i, xds := 0, desc.Extensions(); i < xds.Len(); i++ {
		f.Extensions = append(f.Extensions, newField(gen, f, nil, xds.Get(i)))
	}
	for i, sds := 0, desc.Services(); i < sds.Len(); i++ {
		f.Services = append(f.Services, newService(gen, f, sds.Get(i)))
	}
	for _, message := range f.Messages {
		if err := message.resolveDependencies(gen); err != nil {
			return nil, err
		}
	}
	for _, extension := range f.Extensions {
		if err := extension.resolveDependencies(gen); err != nil {
			return nil, err
		}
	}
	for _, service := range f.Services {
		for _, method := range service.Methods {
			if err := method.resolveDependencies(gen); err != nil {
				return nil, err
			}
		}
	}
	return f, nil
}

// splitImportPathAndPackageName splits off the optional Go package name
// from the Go import path when separated by a ';' delimiter.
func splitImportPathAndPackageName(s string) (GoImportPath, GoPackageName) {
	if i := strings.Index(s, ";"); i >= 0 {
		return GoImportPath(s[:i]), GoPackageName(s[i+1:])
	}
	return GoImportPath(s), ""
}

// An Enum describes an enum.
type Enum struct {
	Desc protoreflect.EnumDescriptor

	GoIdent GoIdent // name of the generated Go type

	Values []*EnumValue // enum value declarations

	Location Location   // location of this enum
	Comments CommentSet // comments associated with this enum
}

func newEnum(gen *Plugin, f *File, parent *Message, desc protoreflect.EnumDescriptor) *Enum {
	var loc Location
	if parent != nil {
		loc = parent.Location.appendPath(genid.DescriptorProto_EnumType_field_number, desc.Index())
	} else {
		loc = f.location.appendPath(genid.FileDescriptorProto_EnumType_field_number, desc.Index())
	}
	enum := &Enum{
		Desc:     desc,
		GoIdent:  newGoIdent(f, desc),
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
	gen.enumsByName[desc.FullName()] = enum
	for i, vds := 0, enum.Desc.Values(); i < vds.Len(); i++ {
		enum.Values = append(enum.Values, newEnumValue(gen, f, parent, enum, vds.Get(i)))
	}
	return enum
}

// An EnumValue describes an enum value.
type EnumValue struct {
	Desc protoreflect.EnumValueDescriptor

	GoIdent GoIdent // name of the generated Go declaration

	Parent *Enum // enum in which this value is declared

	Location Location   // location of this enum value
	Comments CommentSet // comments associated with this enum value
}

func newEnumValue(gen *Plugin, f *File, message *Message, enum *Enum, desc protoreflect.EnumValueDescriptor) *EnumValue {
	// A top-level enum value's name is: EnumName_ValueName
	// An enum value contained in a message is: MessageName_ValueName
	//
	// For historical reasons, enum value names are not camel-cased.
	parentIdent := enum.GoIdent
	if message != nil {
		parentIdent = message.GoIdent
	}
	name := parentIdent.GoName + "_" + string(desc.Name())
	loc := enum.Location.appendPath(genid.EnumDescriptorProto_Value_field_number, desc.Index())
	return &EnumValue{
		Desc:     desc,
		GoIdent:  f.GoImportPath.Ident(name),
		Parent:   enum,
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
}

// A Message describes a message.
type Message struct {
	Desc protoreflect.MessageDescriptor

	GoIdent GoIdent // name of the generated Go type

	Fields []*Field // message field declarations
	Oneofs []*Oneof // message oneof declarations

	Enums      []*Enum      // nested enum declarations
	Messages   []*Message   // nested message declarations
	Extensions []*Extension // nested extension declarations

	Location Location   // location of this message
	Comments CommentSet // comments associated with this message
}

func newMessage(gen *Plugin, f *File, parent *Message, desc protoreflect.MessageDescriptor) *Message {
	var loc Location
	if parent != nil {
		loc = parent.Location.appendPath(genid.DescriptorProto_NestedType_field_number, desc.Index())
	} else {
		loc = f.location.appendPath(genid.FileDescriptorProto_MessageType_field_number, desc.Index())
	}
	message := &Message{
		Desc:     desc,
		GoIdent:  newGoIdent(f, desc),
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
	gen.messagesByName[desc.FullName()] = message
	for i, eds := 0, desc.Enums(); i < eds.Len(); i++ {
		message.Enums = append(message.Enums, newEnum(gen, f, message, eds.Get(i)))
	}
	for i, mds := 0, desc.Messages(); i < mds.Len(); i++ {
		message.Messages = append(message.Messages, newMessage(gen, f, message, mds.Get(i)))
	}
	for i, fds := 0, desc.Fields(); i < fds.Len(); i++ {
		message.Fields = append(message.Fields, newField(gen, f, message, fds.Get(i)))
	}
	for i, ods := 0, desc.Oneofs(); i < ods.Len(); i++ {
		message.Oneofs = append(message.Oneofs, newOneof(gen, f, message, ods.Get(i)))
	}
	for i, xds := 0, desc.Extensions(); i < xds.Len(); i++ {
		message.Extensions = append(message.Extensions, newField(gen, f, message, xds.Get(i)))
	}

	// Resolve local references between fields and oneofs.
	for _, field := range message.Fields {
		if od := field.Desc.ContainingOneof(); od != nil {
			oneof := message.Oneofs[od.Index()]
			field.Oneof = oneof
			oneof.Fields = append(oneof.Fields, field)
		}
	}

	// Field name conflict resolution.
	//
	// We assume well-known method names that may be attached to a generated
	// message type, as well as a 'Get*' method for each field. For each
	// field in turn, we add _s to its name until there are no conflicts.
	//
	// Any change to the following set of method names is a potential
	// incompatible API change because it may change generated field names.
	//
	// TODO: If we ever support a 'go_name' option to set the Go name of a
	// field, we should consider dropping this entirely. The conflict
	// resolution algorithm is subtle and surprising (changing the order
	// in which fields appear in the .proto source file can change the
	// names of fields in generated code), and does not adapt well to
	// adding new per-field methods such as setters.
	usedNames := map[string]bool{
		"Reset":               true,
		"String":              true,
		"ProtoMessage":        true,
		"Marshal":             true,
		"Unmarshal":           true,
		"ExtensionRangeArray": true,
		"ExtensionMap":        true,
		"Descriptor":          true,
	}
	makeNameUnique := func(name string, hasGetter bool) string {
		for usedNames[name] || (hasGetter && usedNames["Get"+name]) {
			name += "_"
		}
		usedNames[name] = true
		usedNames["Get"+name] = hasGetter
		return name
	}
	for _, field := range message.Fields {
		field.GoName = makeNameUnique(field.GoName, true)
		field.GoIdent.GoName = message.GoIdent.GoName + "_" + field.GoName
		if field.Oneof != nil && field.Oneof.Fields[0] == field {
			// Make the name for a oneof unique as well. For historical reasons,
			// this assumes that a getter method is not generated for oneofs.
			// This is incorrect, but fixing it breaks existing code.
			field.Oneof.GoName = makeNameUnique(field.Oneof.GoName, false)
			field.Oneof.GoIdent.GoName = message.GoIdent.GoName + "_" + field.Oneof.GoName
		}
	}

	// Oneof field name conflict resolution.
	//
	// This conflict resolution is incomplete as it does not consider collisions
	// with other oneof field types, but fixing it breaks existing code.
	for _, field := range message.Fields {
		if field.Oneof != nil {
		Loop:
			for {
				for _, nestedMessage := range message.Messages {
					if nestedMessage.GoIdent == field.GoIdent {
						field.GoIdent.GoName += "_"
						continue Loop
					}
				}
				for _, nestedEnum := range message.Enums {
					if nestedEnum.GoIdent == field.GoIdent {
						field.GoIdent.GoName += "_"
						continue Loop
					}
				}
				break Loop
			}
		}
	}

	return message
}

func (message *Message) resolveDependencies(gen *Plugin) error {
	for _, field := range message.Fields {
		if err := field.resolveDependencies(gen); err != nil {
			return err
		}
	}
	for _, message := range message.Messages {
		if err := message.resolveDependencies(gen); err != nil {
			return err
		}
	}
	for _, extension := range message.Extensions {
		if err := extension.resolveDependencies(gen); err != nil {
			return err
		}
	}
	return nil
}

// A Field describes a message field.
type Field struct {
	Desc protoreflect.FieldDescriptor

	// GoName is the base name of this field's Go field and methods.
	// For code generated by protoc-gen-go, this means a field named
	// '{{GoName}}' and a getter method named 'Get{{GoName}}'.
	GoName string // e.g., "FieldName"

	// GoIdent is the base name of a top-level declaration for this field.
	// For code generated by protoc-gen-go, this means a wrapper type named
	// '{{GoIdent}}' for members fields of a oneof, and a variable named
	// 'E_{{GoIdent}}' for extension fields.
	GoIdent GoIdent // e.g., "MessageName_FieldName"

	Parent   *Message // message in which this field is declared; nil if top-level extension
	Oneof    *Oneof   // containing oneof; nil if not part of a oneof
	Extendee *Message // extended message for extension fields; nil otherwise

	Enum    *Enum    // type for enum fields; nil otherwise
	Message *Message // type for message or group fields; nil otherwise

	Location Location   // location of this field
	Comments CommentSet // comments associated with this field
}

func newField(gen *Plugin, f *File, message *Message, desc protoreflect.FieldDescriptor) *Field {
	var loc Location
	switch {
	case desc.IsExtension() && message == nil:
		loc = f.location.appendPath(genid.FileDescriptorProto_Extension_field_number, desc.Index())
	case desc.IsExtension() && message != nil:
		loc = message.Location.appendPath(genid.DescriptorProto_Extension_field_number, desc.Index())
	default:
		loc = message.Location.appendPath(genid.DescriptorProto_Field_field_number, desc.Index())
	}
	camelCased := strs.GoCamelCase(string(desc.Name()))
	var parentPrefix string
	if message != nil {
		parentPrefix = message.GoIdent.GoName + "_"
	}
	field := &Field{
		Desc:   desc,
		GoName: camelCased,
		GoIdent: GoIdent{
			GoImportPath: f.GoImportPath,
			GoName:       parentPrefix + camelCased,
		},
		Parent:   message,
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
	return field
}

func (field *Field) resolveDependencies(gen *Plugin) error {
	desc := field.Desc
	switch desc.Kind() {
	case protoreflect.EnumKind:
		name := field.Desc.Enum().FullName()
		enum, ok := gen.enumsByName[name]
		if !ok {
			return fmt.Errorf("field %v: no descriptor for enum %v", desc.FullName(), name)
		}
		field.Enum = enum
	case protoreflect.MessageKind, protoreflect.GroupKind:
		name := desc.Message().FullName()
		message, ok := gen.messagesByName[name]
		if !ok {
			return fmt.Errorf("field %v: no descriptor for type %v", desc.FullName(), name)
		}
		field.Message = message
	}
	if desc.IsExtension() {
		name := desc.ContainingMessage().FullName()
		message, ok := gen.messagesByName[name]
		if !ok {
			return fmt.Errorf("field %v: no descriptor for type %v", desc.FullName(), name)
		}
		field.Extendee = message
	}
	return nil
}

// A Oneof describes a message oneof.
type Oneof struct {
	Desc protoreflect.OneofDescriptor

	// GoName is the base name of this oneof's Go field and methods.
	// For code generated by protoc-gen-go, this means a field named
	// '{{GoName}}' and a getter method named 'Get{{GoName}}'.
	GoName string // e.g., "OneofName"

	// GoIdent is the base name of a top-level declaration for this oneof.
	GoIdent GoIdent // e.g., "MessageName_OneofName"

	Parent *Message // message in which this oneof is declared

	Fields []*Field // fields that are part of this oneof

	Location Location   // location of this oneof
	Comments CommentSet // comments associated with this oneof
}

func newOneof(gen *Plugin, f *File, message *Message, desc protoreflect.OneofDescriptor) *Oneof {
	loc := message.Location.appendPath(genid.DescriptorProto_OneofDecl_field_number, desc.Index())
	camelCased := strs.GoCamelCase(string(desc.Name()))
	parentPrefix := message.GoIdent.GoName + "_"
	return &Oneof{
		Desc:   desc,
		Parent: message,
		GoName: camelCased,
		GoIdent: GoIdent{
			GoImportPath: f.GoImportPath,
			GoName:       parentPrefix + camelCased,
		},
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
}

// Extension is an alias of [Field] for documentation.
type Extension = Field

// A Service describes a service.
type Service struct {
	Desc protoreflect.ServiceDescriptor

	GoName string

	Methods []*Method // service method declarations

	Location Location   // location of this service
	Comments CommentSet // comments associated with this service
}

func newService(gen *Plugin, f *File, desc protoreflect.ServiceDescriptor) *Service {
	loc := f.location.appendPath(genid.FileDescriptorProto_Service_field_number, desc.Index())
	service := &Service{
		Desc:     desc,
		GoName:   strs.GoCamelCase(string(desc.Name())),
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
	for i, mds := 0, desc.Methods(); i < mds.Len(); i++ {
		service.Methods = append(service.Methods, newMethod(gen, f, service, mds.Get(i)))
	}
	return service
}

// A Method describes a method in a service.
type Method struct {
	Desc protoreflect.MethodDescriptor

	GoName string

	Parent *Service // service in which this method is declared

	Input  *Message
	Output *Message

	Location Location   // location of this method
	Comments CommentSet // comments associated with this method
}

func newMethod(gen *Plugin, f *File, service *Service, desc protoreflect.MethodDescriptor) *Method {
	loc := service.Location.appendPath(genid.ServiceDescriptorProto_Method_field_number, desc.Index())
	method := &Method{
		Desc:     desc,
		GoName:   strs.GoCamelCase(string(desc.Name())),
		Parent:   service,
		Location: loc,
		Comments: makeCommentSet(f.Desc.SourceLocations().ByDescriptor(desc)),
	}
	return method
}

func (method *Method) resolveDependencies(gen *Plugin) error {
	desc := method.Desc

	inName := desc.Input().FullName()
	in, ok := gen.messagesByName[inName]
	if !ok {
		return fmt.Errorf("method %v: no descriptor for type %v", desc.FullName(), inName)
	}
	method.Input = in

	outName := desc.Output().FullName()
	out, ok := gen.messagesByName[outName]
	if !ok {
		return fmt.Errorf("method %v: no descriptor for type %v", desc.FullName(), outName)
	}
	method.Output = out

	return nil
}

// A GeneratedFile is a generated file.
type GeneratedFile struct {
	gen              *Plugin
	skip             bool
	filename         string
	goImportPath     GoImportPath
	buf              bytes.Buffer
	packageNames     map[GoImportPath]GoPackageName
	usedPackageNames map[GoPackageName]bool
	manualImports    map[GoImportPath]bool
	annotations      map[string][]Annotation
}

// NewGeneratedFile creates a new generated file with the given filename
// and import path.
func (gen *Plugin) NewGeneratedFile(filename string, goImportPath GoImportPath) *GeneratedFile {
	g := &GeneratedFile{
		gen:              gen,
		filename:         filename,
		goImportPath:     goImportPath,
		packageNames:     make(map[GoImportPath]GoPackageName),
		usedPackageNames: make(map[GoPackageName]bool),
		manualImports:    make(map[GoImportPath]bool),
		annotations:      make(map[string][]Annotation),
	}

	// All predeclared identifiers in Go are already used.
	for _, s := range types.Universe.Names() {
		g.usedPackageNames[GoPackageName(s)] = true
	}

	gen.genFiles = append(gen.genFiles, g)
	return g
}

// P prints a line to the generated output. It converts each parameter to a
// string following the same rules as [fmt.Print]. It never inserts spaces
// between parameters.
func (g *GeneratedFile) P(v ...interface{}) {
	for _, x := range v {
		switch x := x.(type) {
		case GoIdent:
			fmt.Fprint(&g.buf, g.QualifiedGoIdent(x))
		default:
			fmt.Fprint(&g.buf, x)
		}
	}
	fmt.Fprintln(&g.buf)
}

// QualifiedGoIdent returns the string to use for a Go identifier.
//
// If the identifier is from a different Go package than the generated file,
// the returned name will be qualified (package.name) and an import statement
// for the identifier's package will be included in the file.
func (g *GeneratedFile) QualifiedGoIdent(ident GoIdent) string {
	if ident.GoImportPath == g.goImportPath {
		return ident.GoName
	}
	if packageName, ok := g.packageNames[ident.GoImportPath]; ok {
		return string(packageName) + "." + ident.GoName
	}
	packageName := cleanPackageName(path.Base(string(ident.GoImportPath)))
	for i, orig := 1, packageName; g.usedPackageNames[packageName]; i++ {
		packageName = orig + GoPackageName(strconv.Itoa(i))
	}
	g.packageNames[ident.GoImportPath] = packageName
	g.usedPackageNames[packageName] = true
	return string(packageName) + "." + ident.GoName
}

// Import ensures a package is imported by the generated file.
//
// Packages referenced by [GeneratedFile.QualifiedGoIdent] are automatically imported.
// Explicitly importing a package with Import is generally only necessary
// when the import will be blank (import _ "package").
func (g *GeneratedFile) Import(importPath GoImportPath) {
	g.manualImports[importPath] = true
}

// Write implements [io.Writer].
func (g *GeneratedFile) Write(p []byte) (n int, err error) {
	return g.buf.Write(p)
}

// Skip removes the generated file from the plugin output.
func (g *GeneratedFile) Skip() {
	g.skip = true
}

// Unskip reverts a previous call to [GeneratedFile.Skip],
// re-including the generated file in the plugin output.
func (g *GeneratedFile) Unskip() {
	g.skip = false
}

// Annotate associates a symbol in a generated Go file with a location in a
// source .proto file.
//
// The symbol may refer to a type, constant, variable, function, method, or
// struct field.  The "T.sel" syntax is used to identify the method or field
// 'sel' on type 'T'.
//
// Deprecated: Use the [GeneratedFile.AnnotateSymbol] method instead.
func (g *GeneratedFile) Annotate(symbol string, loc Location) {
	g.AnnotateSymbol(symbol, Annotation{Location: loc})
}

// An Annotation provides semantic detail for a generated proto element.
//
// See the google.protobuf.GeneratedCodeInfo.Annotation documentation in
// descriptor.proto for details.
type Annotation struct {
	// Location is the source .proto file for the element.
	Location Location

	// Semantic is the symbol's effect on the element in the original .proto file.
	Semantic *descriptorpb.GeneratedCodeInfo_Annotation_Semantic
}

// AnnotateSymbol associates a symbol in a generated Go file with a location
// in a source .proto file and a semantic type.
//
// The symbol may refer to a type, constant, variable, function, method, or
// struct field.  The "T.sel" syntax is used to identify the method or field
// 'sel' on type 'T'.
func (g *GeneratedFile) AnnotateSymbol(symbol string, info Annotation) {
	g.annotations[symbol] = append(g.annotations[symbol], info)
}

// Content returns the contents of the generated file.
func (g *GeneratedFile) Content() ([]byte, error) {
	if !strings.HasSuffix(g.filename, ".go") {
		return g.buf.Bytes(), nil
	}

	// Reformat generated code.
	original := g.buf.Bytes()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", original, parser.ParseComments)
	if err != nil {
		// Print out the bad code with line numbers.
		// This should never happen in practice, but it can while changing generated code
		// so consider this a debugging aid.
		var src bytes.Buffer
		s := bufio.NewScanner(bytes.NewReader(original))
		for line := 1; s.Scan(); line++ {
			fmt.Fprintf(&src, "%5d\t%s\n", line, s.Bytes())
		}
		return nil, fmt.Errorf("%v: unparsable Go source: %v\n%v", g.filename, err, src.String())
	}

	// Collect a sorted list of all imports.
	var importPaths [][2]string
	rewriteImport := func(importPath string) string {
		if f := g.gen.opts.ImportRewriteFunc; f != nil {
			return string(f(GoImportPath(importPath)))
		}
		return importPath
	}
	for importPath := range g.packageNames {
		pkgName := string(g.packageNames[GoImportPath(importPath)])
		pkgPath := rewriteImport(string(importPath))
		importPaths = append(importPaths, [2]string{pkgName, pkgPath})
	}
	for importPath := range g.manualImports {
		if _, ok := g.packageNames[importPath]; !ok {
			pkgPath := rewriteImport(string(importPath))
			importPaths = append(importPaths, [2]string{"_", pkgPath})
		}
	}
	sort.Slice(importPaths, func(i, j int) bool {
		return importPaths[i][1] < importPaths[j][1]
	})

	// Modify the AST to include a new import block.
	if len(importPaths) > 0 {
		// Insert block after package statement or
		// possible comment attached to the end of the package statement.
		pos := file.Package
		tokFile := fset.File(file.Package)
		pkgLine := tokFile.Line(file.Package)
		for _, c := range file.Comments {
			if tokFile.Line(c.Pos()) > pkgLine {
				break
			}
			pos = c.End()
		}

		// Construct the import block.
		impDecl := &ast.GenDecl{
			Tok:    token.IMPORT,
			TokPos: pos,
			Lparen: pos,
			Rparen: pos,
		}
		for _, importPath := range importPaths {
			impDecl.Specs = append(impDecl.Specs, &ast.ImportSpec{
				Name: &ast.Ident{
					Name:    importPath[0],
					NamePos: pos,
				},
				Path: &ast.BasicLit{
					Kind:     token.STRING,
					Value:    strconv.Quote(importPath[1]),
					ValuePos: pos,
				},
				EndPos: pos,
			})
		}
		file.Decls = append([]ast.Decl{impDecl}, file.Decls...)
	}

	var out bytes.Buffer
	if err = (&printer.Config{Mode: printer.TabIndent | printer.UseSpaces, Tabwidth: 8}).Fprint(&out, fset, file); err != nil {
		return nil, fmt.Errorf("%v: can not reformat Go source: %v", g.filename, err)
	}
	return out.Bytes(), nil
}

func (g *GeneratedFile) generatedCodeInfo(content []byte) (*descriptorpb.GeneratedCodeInfo, error) {
	fset := token.NewFileSet()
	astFile, err := parser.ParseFile(fset, "", content, 0)
	if err != nil {
		return nil, err
	}
	info := &descriptorpb.GeneratedCodeInfo{}

	seenAnnotations := make(map[string]bool)
	annotate := func(s string, ident *ast.Ident) {
		seenAnnotations[s] = true
		for _, a := range g.annotations[s] {
			info.Annotation = append(info.Annotation, &descriptorpb.GeneratedCodeInfo_Annotation{
				SourceFile: proto.String(a.Location.SourceFile),
				Path:       a.Location.Path,
				Begin:      proto.Int32(int32(fset.Position(ident.Pos()).Offset)),
				End:        proto.Int32(int32(fset.Position(ident.End()).Offset)),
				Semantic:   a.Semantic,
			})
		}
	}
	for _, decl := range astFile.Decls {
		switch decl := decl.(type) {
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.TypeSpec:
					annotate(spec.Name.Name, spec.Name)
					switch st := spec.Type.(type) {
					case *ast.StructType:
						for _, field := range st.Fields.List {
							for _, name := range field.Names {
								annotate(spec.Name.Name+"."+name.Name, name)
							}
						}
					case *ast.InterfaceType:
						for _, field := range st.Methods.List {
							for _, name := range field.Names {
								annotate(spec.Name.Name+"."+name.Name, name)
							}
						}
					}
				case *ast.ValueSpec:
					for _, name := range spec.Names {
						annotate(name.Name, name)
					}
				}
			}
		case *ast.FuncDecl:
			if decl.Recv == nil {
				annotate(decl.Name.Name, decl.Name)
			} else {
				recv := decl.Recv.List[0].Type
				if s, ok := recv.(*ast.StarExpr); ok {
					recv = s.X
				}
				if id, ok := recv.(*ast.Ident); ok {
					annotate(id.Name+"."+decl.Name.Name, decl.Name)
				}
			}
		}
	}
	for a := range g.annotations {
		if !seenAnnotations[a] {
			return nil, fmt.Errorf("%v: no symbol matching annotation %q", g.filename, a)
		}
	}

	return info, nil
}

// metaFile returns the contents of the file's metadata file, which is a
// text formatted string of the google.protobuf.GeneratedCodeInfo.
func (g *GeneratedFile) metaFile(content []byte) (string, error) {
	info, err := g.generatedCodeInfo(content)
	if err != nil {
		return "", err
	}

	b, err := prototext.Marshal(info)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// A GoIdent is a Go identifier, consisting of a name and import path.
// The name is a single identifier and may not be a dot-qualified selector.
type GoIdent struct {
	GoName       string
	GoImportPath GoImportPath
}

func (id GoIdent) String() string { return fmt.Sprintf("%q.%v", id.GoImportPath, id.GoName) }

// newGoIdent returns the Go identifier for a descriptor.
func newGoIdent(f *File, d protoreflect.Descriptor) GoIdent {
	name := strings.TrimPrefix(string(d.FullName()), string(f.Desc.Package())+".")
	return GoIdent{
		GoName:       strs.GoCamelCase(name),
		GoImportPath: f.GoImportPath,
	}
}

// A GoImportPath is the import path of a Go package.
// For example: "google.golang.org/protobuf/compiler/protogen"
type GoImportPath string

func (p GoImportPath) String() string { return strconv.Quote(string(p)) }

// Ident returns a GoIdent with s as the GoName and p as the GoImportPath.
func (p GoImportPath) Ident(s string) GoIdent {
	return GoIdent{GoName: s, GoImportPath: p}
}

// A GoPackageName is the name of a Go package. e.g., "protobuf".
type GoPackageName string

// cleanPackageName converts a string to a valid Go package name.
func cleanPackageName(name string) GoPackageName {
	return GoPackageName(strs.GoSanitized(name))
}

type pathType int

const (
	pathTypeImport pathType = iota
	pathTypeSourceRelative
)

// A Location is a location in a .proto source file.
//
// See the google.protobuf.SourceCodeInfo documentation in descriptor.proto
// for details.
type Location struct {
	SourceFile string
	Path       protoreflect.SourcePath
}

// appendPath add elements to a Location's path, returning a new Location.
func (loc Location) appendPath(num protoreflect.FieldNumber, idx int) Location {
	loc.Path = append(protoreflect.SourcePath(nil), loc.Path...) // make copy
	loc.Path = append(loc.Path, int32(num), int32(idx))
	return loc
}

// CommentSet is a set of leading and trailing comments associated
// with a .proto descriptor declaration.
type CommentSet struct {
	LeadingDetached []Comments
	Leading         Comments
	Trailing        Comments
}

func makeCommentSet(loc protoreflect.SourceLocation) CommentSet {
	var leadingDetached []Comments
	for _, s := range loc.LeadingDetachedComments {
		leadingDetached = append(leadingDetached, Comments(s))
	}
	return CommentSet{
		LeadingDetached: leadingDetached,
		Leading:         Comments(loc.LeadingComments),
		Trailing:        Comments(loc.TrailingComments),
	}
}

// Comments is a comments string as provided by protoc.
type Comments string

// String formats the comments by inserting // to the start of each line,
// ensuring that there is a trailing newline.
// An empty comment is formatted as an empty string.
func (c Comments) String() string {
	if c == "" {
		return ""
	}
	var b []byte
	for _, line := range strings.Split(strings.TrimSuffix(string(c), "\n"), "\n") {
		b = append(b, "//"...)
		b = append(b, line...)
		b = append(b, "\n"...)
	}
	return string(b)
}

// extensionRegistry allows registration of new extensions defined in the .proto
// file for which we are generating bindings.
//
// Lookups consult the local type registry first and fall back to the base type
// registry which defaults to protoregistry.GlobalTypes.
type extensionRegistry struct {
	base  *protoregistry.Types
	local *protoregistry.Types
}

func newExtensionRegistry() *extensionRegistry {
	return &extensionRegistry{
		base:  protoregistry.GlobalTypes,
		local: &protoregistry.Types{},
	}
}

// FindExtensionByName implements proto.UnmarshalOptions.FindExtensionByName
func (e *extensionRegistry) FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error) {
	if xt, err := e.local.FindExtensionByName(field); err == nil {
		return xt, nil
	}

	return e.base.FindExtensionByName(field)
}

// FindExtensionByNumber implements proto.UnmarshalOptions.FindExtensionByNumber
func (e *extensionRegistry) FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error) {
	if xt, err := e.local.FindExtensionByNumber(message, field); err == nil {
		return xt, nil
	}

	return e.base.FindExtensionByNumber(message, field)
}

func (e *extensionRegistry) hasNovelExtensions() bool {
	return e.local.NumExtensions() > 0
}

func (e *extensionRegistry) registerAllExtensionsFromFile(f protoreflect.FileDescriptor) error {
	if err := e.registerAllExtensions(f.Extensions()); err != nil {
		return err
	}
	return nil
}

func (e *extensionRegistry) registerAllExtensionsFromMessage(ms protoreflect.MessageDescriptors) error {
	for i := 0; i < ms.Len(); i++ {
		m := ms.Get(i)
		if err := e.registerAllExtensions(m.Extensions()); err != nil {
			return err
		}
	}
	return nil
}

func (e *extensionRegistry) registerAllExtensions(exts protoreflect.ExtensionDescriptors) error {
	for i := 0; i < exts.Len(); i++ {
		if err := e.registerExtension(exts.Get(i)); err != nil {
			return err
		}
	}
	return nil
}

// registerExtension adds the given extension to the type registry if an
// extension with that full name does not exist yet.
func (e *extensionRegistry) registerExtension(xd protoreflect.ExtensionDescriptor) error {
	if _, err := e.FindExtensionByName(xd.FullName()); err != protoregistry.NotFound {
		// Either the extension already exists or there was an error, either way we're done.
		return err
	}
	return e.local.RegisterExtension(dynamicpb.NewExtensionType(xd))
}
