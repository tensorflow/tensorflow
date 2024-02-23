// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protodesc provides functionality for converting
// FileDescriptorProto messages to/from [protoreflect.FileDescriptor] values.
//
// The google.protobuf.FileDescriptorProto is a protobuf message that describes
// the type information for a .proto file in a form that is easily serializable.
// The [protoreflect.FileDescriptor] is a more structured representation of
// the FileDescriptorProto message where references and remote dependencies
// can be directly followed.
package protodesc

import (
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	"google.golang.org/protobuf/types/descriptorpb"
)

// Resolver is the resolver used by [NewFile] to resolve dependencies.
// The enums and messages provided must belong to some parent file,
// which is also registered.
//
// It is implemented by [protoregistry.Files].
type Resolver interface {
	FindFileByPath(string) (protoreflect.FileDescriptor, error)
	FindDescriptorByName(protoreflect.FullName) (protoreflect.Descriptor, error)
}

// FileOptions configures the construction of file descriptors.
type FileOptions struct {
	pragma.NoUnkeyedLiterals

	// AllowUnresolvable configures New to permissively allow unresolvable
	// file, enum, or message dependencies. Unresolved dependencies are replaced
	// by placeholder equivalents.
	//
	// The following dependencies may be left unresolved:
	//	• Resolving an imported file.
	//	• Resolving the type for a message field or extension field.
	//	If the kind of the field is unknown, then a placeholder is used for both
	//	the Enum and Message accessors on the protoreflect.FieldDescriptor.
	//	• Resolving an enum value set as the default for an optional enum field.
	//	If unresolvable, the protoreflect.FieldDescriptor.Default is set to the
	//	first value in the associated enum (or zero if the also enum dependency
	//	is also unresolvable). The protoreflect.FieldDescriptor.DefaultEnumValue
	//	is populated with a placeholder.
	//	• Resolving the extended message type for an extension field.
	//	• Resolving the input or output message type for a service method.
	//
	// If the unresolved dependency uses a relative name,
	// then the placeholder will contain an invalid FullName with a "*." prefix,
	// indicating that the starting prefix of the full name is unknown.
	AllowUnresolvable bool
}

// NewFile creates a new [protoreflect.FileDescriptor] from the provided
// file descriptor message. See [FileOptions.New] for more information.
func NewFile(fd *descriptorpb.FileDescriptorProto, r Resolver) (protoreflect.FileDescriptor, error) {
	return FileOptions{}.New(fd, r)
}

// NewFiles creates a new [protoregistry.Files] from the provided
// FileDescriptorSet message. See [FileOptions.NewFiles] for more information.
func NewFiles(fd *descriptorpb.FileDescriptorSet) (*protoregistry.Files, error) {
	return FileOptions{}.NewFiles(fd)
}

// New creates a new [protoreflect.FileDescriptor] from the provided
// file descriptor message. The file must represent a valid proto file according
// to protobuf semantics. The returned descriptor is a deep copy of the input.
//
// Any imported files, enum types, or message types referenced in the file are
// resolved using the provided registry. When looking up an import file path,
// the path must be unique. The newly created file descriptor is not registered
// back into the provided file registry.
func (o FileOptions) New(fd *descriptorpb.FileDescriptorProto, r Resolver) (protoreflect.FileDescriptor, error) {
	if r == nil {
		r = (*protoregistry.Files)(nil) // empty resolver
	}

	// Handle the file descriptor content.
	f := &filedesc.File{L2: &filedesc.FileL2{}}
	switch fd.GetSyntax() {
	case "proto2", "":
		f.L1.Syntax = protoreflect.Proto2
	case "proto3":
		f.L1.Syntax = protoreflect.Proto3
	case "editions":
		f.L1.Syntax = protoreflect.Editions
		f.L1.Edition = fromEditionProto(fd.GetEdition())
	default:
		return nil, errors.New("invalid syntax: %q", fd.GetSyntax())
	}
	if f.L1.Syntax == protoreflect.Editions && (fd.GetEdition() < SupportedEditionsMinimum || fd.GetEdition() > SupportedEditionsMaximum) {
		return nil, errors.New("use of edition %v not yet supported by the Go Protobuf runtime", fd.GetEdition())
	}
	f.L1.Path = fd.GetName()
	if f.L1.Path == "" {
		return nil, errors.New("file path must be populated")
	}
	f.L1.Package = protoreflect.FullName(fd.GetPackage())
	if !f.L1.Package.IsValid() && f.L1.Package != "" {
		return nil, errors.New("invalid package: %q", f.L1.Package)
	}
	if opts := fd.GetOptions(); opts != nil {
		opts = proto.Clone(opts).(*descriptorpb.FileOptions)
		f.L2.Options = func() protoreflect.ProtoMessage { return opts }
	}
	if f.L1.Syntax == protoreflect.Editions {
		initFileDescFromFeatureSet(f, fd.GetOptions().GetFeatures())
	}

	f.L2.Imports = make(filedesc.FileImports, len(fd.GetDependency()))
	for _, i := range fd.GetPublicDependency() {
		if !(0 <= i && int(i) < len(f.L2.Imports)) || f.L2.Imports[i].IsPublic {
			return nil, errors.New("invalid or duplicate public import index: %d", i)
		}
		f.L2.Imports[i].IsPublic = true
	}
	for _, i := range fd.GetWeakDependency() {
		if !(0 <= i && int(i) < len(f.L2.Imports)) || f.L2.Imports[i].IsWeak {
			return nil, errors.New("invalid or duplicate weak import index: %d", i)
		}
		f.L2.Imports[i].IsWeak = true
	}
	imps := importSet{f.Path(): true}
	for i, path := range fd.GetDependency() {
		imp := &f.L2.Imports[i]
		f, err := r.FindFileByPath(path)
		if err == protoregistry.NotFound && (o.AllowUnresolvable || imp.IsWeak) {
			f = filedesc.PlaceholderFile(path)
		} else if err != nil {
			return nil, errors.New("could not resolve import %q: %v", path, err)
		}
		imp.FileDescriptor = f

		if imps[imp.Path()] {
			return nil, errors.New("already imported %q", path)
		}
		imps[imp.Path()] = true
	}
	for i := range fd.GetDependency() {
		imp := &f.L2.Imports[i]
		imps.importPublic(imp.Imports())
	}

	// Handle source locations.
	f.L2.Locations.File = f
	for _, loc := range fd.GetSourceCodeInfo().GetLocation() {
		var l protoreflect.SourceLocation
		// TODO: Validate that the path points to an actual declaration?
		l.Path = protoreflect.SourcePath(loc.GetPath())
		s := loc.GetSpan()
		switch len(s) {
		case 3:
			l.StartLine, l.StartColumn, l.EndLine, l.EndColumn = int(s[0]), int(s[1]), int(s[0]), int(s[2])
		case 4:
			l.StartLine, l.StartColumn, l.EndLine, l.EndColumn = int(s[0]), int(s[1]), int(s[2]), int(s[3])
		default:
			return nil, errors.New("invalid span: %v", s)
		}
		// TODO: Validate that the span information is sensible?
		// See https://github.com/protocolbuffers/protobuf/issues/6378.
		if false && (l.EndLine < l.StartLine || l.StartLine < 0 || l.StartColumn < 0 || l.EndColumn < 0 ||
			(l.StartLine == l.EndLine && l.EndColumn <= l.StartColumn)) {
			return nil, errors.New("invalid span: %v", s)
		}
		l.LeadingDetachedComments = loc.GetLeadingDetachedComments()
		l.LeadingComments = loc.GetLeadingComments()
		l.TrailingComments = loc.GetTrailingComments()
		f.L2.Locations.List = append(f.L2.Locations.List, l)
	}

	// Step 1: Allocate and derive the names for all declarations.
	// This copies all fields from the descriptor proto except:
	//	google.protobuf.FieldDescriptorProto.type_name
	//	google.protobuf.FieldDescriptorProto.default_value
	//	google.protobuf.FieldDescriptorProto.oneof_index
	//	google.protobuf.FieldDescriptorProto.extendee
	//	google.protobuf.MethodDescriptorProto.input
	//	google.protobuf.MethodDescriptorProto.output
	var err error
	sb := new(strs.Builder)
	r1 := make(descsByName)
	if f.L1.Enums.List, err = r1.initEnumDeclarations(fd.GetEnumType(), f, sb); err != nil {
		return nil, err
	}
	if f.L1.Messages.List, err = r1.initMessagesDeclarations(fd.GetMessageType(), f, sb); err != nil {
		return nil, err
	}
	if f.L1.Extensions.List, err = r1.initExtensionDeclarations(fd.GetExtension(), f, sb); err != nil {
		return nil, err
	}
	if f.L1.Services.List, err = r1.initServiceDeclarations(fd.GetService(), f, sb); err != nil {
		return nil, err
	}

	// Step 2: Resolve every dependency reference not handled by step 1.
	r2 := &resolver{local: r1, remote: r, imports: imps, allowUnresolvable: o.AllowUnresolvable}
	if err := r2.resolveMessageDependencies(f.L1.Messages.List, fd.GetMessageType()); err != nil {
		return nil, err
	}
	if err := r2.resolveExtensionDependencies(f.L1.Extensions.List, fd.GetExtension()); err != nil {
		return nil, err
	}
	if err := r2.resolveServiceDependencies(f.L1.Services.List, fd.GetService()); err != nil {
		return nil, err
	}

	// Step 3: Validate every enum, message, and extension declaration.
	if err := validateEnumDeclarations(f.L1.Enums.List, fd.GetEnumType()); err != nil {
		return nil, err
	}
	if err := validateMessageDeclarations(f.L1.Messages.List, fd.GetMessageType()); err != nil {
		return nil, err
	}
	if err := validateExtensionDeclarations(f.L1.Extensions.List, fd.GetExtension()); err != nil {
		return nil, err
	}

	return f, nil
}

type importSet map[string]bool

func (is importSet) importPublic(imps protoreflect.FileImports) {
	for i := 0; i < imps.Len(); i++ {
		if imp := imps.Get(i); imp.IsPublic {
			is[imp.Path()] = true
			is.importPublic(imp.Imports())
		}
	}
}

// NewFiles creates a new [protoregistry.Files] from the provided
// FileDescriptorSet message. The descriptor set must include only
// valid files according to protobuf semantics. The returned descriptors
// are a deep copy of the input.
func (o FileOptions) NewFiles(fds *descriptorpb.FileDescriptorSet) (*protoregistry.Files, error) {
	files := make(map[string]*descriptorpb.FileDescriptorProto)
	for _, fd := range fds.File {
		if _, ok := files[fd.GetName()]; ok {
			return nil, errors.New("file appears multiple times: %q", fd.GetName())
		}
		files[fd.GetName()] = fd
	}
	r := &protoregistry.Files{}
	for _, fd := range files {
		if err := o.addFileDeps(r, fd, files); err != nil {
			return nil, err
		}
	}
	return r, nil
}
func (o FileOptions) addFileDeps(r *protoregistry.Files, fd *descriptorpb.FileDescriptorProto, files map[string]*descriptorpb.FileDescriptorProto) error {
	// Set the entry to nil while descending into a file's dependencies to detect cycles.
	files[fd.GetName()] = nil
	for _, dep := range fd.Dependency {
		depfd, ok := files[dep]
		if depfd == nil {
			if ok {
				return errors.New("import cycle in file: %q", dep)
			}
			continue
		}
		if err := o.addFileDeps(r, depfd, files); err != nil {
			return err
		}
	}
	// Delete the entry once dependencies are processed.
	delete(files, fd.GetName())
	f, err := o.New(fd, r)
	if err != nil {
		return err
	}
	return r.RegisterFile(f)
}
