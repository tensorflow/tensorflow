// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protogen

import (
	"flag"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/testing/protocmp"

	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/pluginpb"
)

func TestPluginParameters(t *testing.T) {
	var flags flag.FlagSet
	value := flags.Int("integer", 0, "")
	const params = "integer=2"
	_, err := Options{
		ParamFunc: flags.Set,
	}.New(&pluginpb.CodeGeneratorRequest{
		Parameter: proto.String(params),
	})
	if err != nil {
		t.Errorf("New(generator parameters %q): %v", params, err)
	}
	if *value != 2 {
		t.Errorf("New(generator parameters %q): integer=%v, want 2", params, *value)
	}
}

func TestPluginParameterErrors(t *testing.T) {
	for _, parameter := range []string{
		"unknown=1",
		"boolean=error",
	} {
		var flags flag.FlagSet
		flags.Bool("boolean", false, "")
		_, err := Options{
			ParamFunc: flags.Set,
		}.New(&pluginpb.CodeGeneratorRequest{
			Parameter: proto.String(parameter),
		})
		if err == nil {
			t.Errorf("New(generator parameters %q): want error, got nil", parameter)
		}
	}
}

func TestNoGoPackage(t *testing.T) {
	_, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("testdata/go_package/no_go_package.proto"),
				Syntax:  proto.String(protoreflect.Proto3.String()),
				Package: proto.String("goproto.testdata"),
			},
		},
	})
	if err == nil {
		t.Fatalf("missing go_package option: New(req) = nil, want error")
	}
}

func TestInvalidImportPath(t *testing.T) {
	_, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("testdata/go_package/no_go_package.proto"),
				Syntax:  proto.String(protoreflect.Proto3.String()),
				Package: proto.String("goproto.testdata"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("foo"),
				},
			},
		},
	})
	if err == nil {
		t.Fatalf("missing go_package option: New(req) = nil, want error")
	}
}

func TestPackageNamesAndPaths(t *testing.T) {
	const (
		filename         = "dir/filename.proto"
		protoPackageName = "proto.package"
	)
	for _, test := range []struct {
		desc            string
		parameter       string
		goPackageOption string
		generate        bool
		wantPackageName GoPackageName
		wantImportPath  GoImportPath
		wantFilename    string
	}{
		{
			desc:            "go_package option sets import path",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "go_package option sets import path without slashes",
			goPackageOption: "golang.org;foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org",
			wantFilename:    "golang.org/filename",
		},
		{
			desc:            "go_package option sets import path and package",
			goPackageOption: "golang.org/x/foo;bar",
			generate:        true,
			wantPackageName: "bar",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "golang.org/x/foo/filename",
		},
		{
			desc:            "command line sets import path for a file",
			parameter:       "Mdir/filename.proto=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/bar/filename",
		},
		{
			desc:            "command line sets import path for a file with package name specified",
			parameter:       "Mdir/filename.proto=golang.org/x/bar;bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "bar",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/bar/filename",
		},
		{
			desc:            "module option set",
			parameter:       "module=golang.org/x",
			goPackageOption: "golang.org/x/foo",
			generate:        false,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "foo/filename",
		},
		{
			desc:            "paths=import uses import path from command line",
			parameter:       "paths=import,Mdir/filename.proto=golang.org/x/bar",
			goPackageOption: "golang.org/x/foo",
			generate:        true,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/bar",
			wantFilename:    "golang.org/x/bar/filename",
		},
		{
			desc:            "module option implies paths=import",
			parameter:       "module=golang.org/x,Mdir/filename.proto=golang.org/x/foo",
			generate:        false,
			wantPackageName: "foo",
			wantImportPath:  "golang.org/x/foo",
			wantFilename:    "foo/filename",
		},
	} {
		context := fmt.Sprintf(`
TEST: %v
  --go_out=%v:.
  file %q: generate=%v
  option go_package = %q;

  `,
			test.desc, test.parameter, filename, test.generate, test.goPackageOption)

		req := &pluginpb.CodeGeneratorRequest{
			Parameter: proto.String(test.parameter),
			ProtoFile: []*descriptorpb.FileDescriptorProto{
				{
					Name:    proto.String(filename),
					Package: proto.String(protoPackageName),
					Options: &descriptorpb.FileOptions{
						GoPackage: proto.String(test.goPackageOption),
					},
				},
			},
		}
		if test.generate {
			req.FileToGenerate = []string{filename}
		}
		gen, err := Options{}.New(req)
		if err != nil {
			t.Errorf("%vNew(req) = %v", context, err)
			continue
		}
		gotFile, ok := gen.FilesByPath[filename]
		if !ok {
			t.Errorf("%v%v: missing file info", context, filename)
			continue
		}
		if got, want := gotFile.GoPackageName, test.wantPackageName; got != want {
			t.Errorf("%vGoPackageName=%v, want %v", context, got, want)
		}
		if got, want := gotFile.GoImportPath, test.wantImportPath; got != want {
			t.Errorf("%vGoImportPath=%v, want %v", context, got, want)
		}
		gen.NewGeneratedFile(gotFile.GeneratedFilenamePrefix, "")
		resp := gen.Response()
		if got, want := resp.File[0].GetName(), test.wantFilename; got != want {
			t.Errorf("%vgenerated filename=%v, want %v", context, got, want)
		}
	}
}

func TestPackageNameInference(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		Parameter: proto.String("Mdir/file1.proto=path/to/file1"),
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("dir/file1.proto"),
				Package: proto.String("proto.package"),
			},
			{
				Name:    proto.String("dir/file2.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("path/to/file2"),
				},
			},
		},
		FileToGenerate: []string{"dir/file1.proto", "dir/file2.proto"},
	})
	if err != nil {
		t.Fatalf("New(req) = %v", err)
	}
	if f1, ok := gen.FilesByPath["dir/file1.proto"]; !ok {
		t.Errorf("missing file info for dir/file1.proto")
	} else if f1.GoPackageName != "file1" {
		t.Errorf("dir/file1.proto: GoPackageName=%v, want foo; package name should be derived from dir/file2.proto", f1.GoPackageName)
	}
}

func TestInconsistentPackageNames(t *testing.T) {
	_, err := Options{}.New(&pluginpb.CodeGeneratorRequest{
		ProtoFile: []*descriptorpb.FileDescriptorProto{
			{
				Name:    proto.String("dir/file1.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("golang.org/x/foo"),
				},
			},
			{
				Name:    proto.String("dir/file2.proto"),
				Package: proto.String("proto.package"),
				Options: &descriptorpb.FileOptions{
					GoPackage: proto.String("golang.org/x/foo;bar"),
				},
			},
		},
		FileToGenerate: []string{"dir/file1.proto", "dir/file2.proto"},
	})
	if err == nil {
		t.Fatalf("inconsistent package names for the same import path: New(req) = nil, want error")
	}
}

func TestImports(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{})
	if err != nil {
		t.Fatal(err)
	}
	g := gen.NewGeneratedFile("foo.go", "golang.org/x/foo")
	g.P("package foo")
	g.P()
	for _, importPath := range []GoImportPath{
		"golang.org/x/foo",
		// Multiple references to the same package.
		"golang.org/x/bar",
		"golang.org/x/bar",
		// Reference to a different package with the same basename.
		"golang.org/y/bar",
		"golang.org/x/baz",
		// Reference to a package conflicting with a predeclared identifier.
		"golang.org/z/string",
	} {
		g.P("var _ = ", GoIdent{GoName: "X", GoImportPath: importPath}, " // ", importPath)
	}
	want := `package foo

import (
	bar "golang.org/x/bar"
	baz "golang.org/x/baz"
	bar1 "golang.org/y/bar"
	string1 "golang.org/z/string"
)

var _ = X         // "golang.org/x/foo"
var _ = bar.X     // "golang.org/x/bar"
var _ = bar.X     // "golang.org/x/bar"
var _ = bar1.X    // "golang.org/y/bar"
var _ = baz.X     // "golang.org/x/baz"
var _ = string1.X // "golang.org/z/string"
`
	got, err := g.Content()
	if err != nil {
		t.Fatalf("g.Content() = %v", err)
	}
	if diff := cmp.Diff(string(want), string(got)); diff != "" {
		t.Fatalf("content mismatch (-want +got):\n%s", diff)
	}
}

func TestImportRewrites(t *testing.T) {
	gen, err := Options{
		ImportRewriteFunc: func(i GoImportPath) GoImportPath {
			return "prefix/" + i
		},
	}.New(&pluginpb.CodeGeneratorRequest{})
	if err != nil {
		t.Fatal(err)
	}
	g := gen.NewGeneratedFile("foo.go", "golang.org/x/foo")
	g.P("package foo")
	g.P("var _ = ", GoIdent{GoName: "X", GoImportPath: "golang.org/x/bar"})
	want := `package foo

import (
	bar "prefix/golang.org/x/bar"
)

var _ = bar.X
`
	got, err := g.Content()
	if err != nil {
		t.Fatalf("g.Content() = %v", err)
	}
	if diff := cmp.Diff(string(want), string(got)); diff != "" {
		t.Fatalf("content mismatch (-want +got):\n%s", diff)
	}
}

func TestAnnotations(t *testing.T) {
	gen, err := Options{}.New(&pluginpb.CodeGeneratorRequest{})
	if err != nil {
		t.Fatal(err)
	}
	loc := Location{SourceFile: "foo.go"}
	g := gen.NewGeneratedFile("foo.go", "golang.org/x/foo")

	g.P("package foo")
	g.P()

	structName := "S"
	fieldName := "Field"

	messageLoc := loc.appendPath(genid.FileDescriptorProto_MessageType_field_number, 1)
	fieldLoc := messageLoc.appendPath(genid.DescriptorProto_Field_field_number, 1)

	g.Annotate(structName, messageLoc) // use deprecated version to test existing usages
	g.P("type ", structName, " struct {")
	g.AnnotateSymbol(structName+"."+fieldName, Annotation{Location: fieldLoc})
	g.P(fieldName, " string")
	g.P("}")
	g.P()

	g.AnnotateSymbol(fmt.Sprintf("%s.Set%s", structName, fieldName), Annotation{
		Location: fieldLoc,
		Semantic: descriptorpb.GeneratedCodeInfo_Annotation_SET.Enum(),
	})
	g.P("func (m *", structName, ") Set", fieldName, "(x string) {")
	g.P("m.", fieldName, " = x")
	g.P("}")
	g.P()

	want := &descriptorpb.GeneratedCodeInfo{
		Annotation: []*descriptorpb.GeneratedCodeInfo_Annotation{
			{ // S
				SourceFile: proto.String("foo.go"),
				Path:       []int32{4, 1}, // message S
				Begin:      proto.Int32(18),
				End:        proto.Int32(19),
			},
			{ // S.F
				SourceFile: proto.String("foo.go"),
				Path:       []int32{4, 1, 2, 1},
				Begin:      proto.Int32(30),
				End:        proto.Int32(35),
			},
			{ // SetF
				SourceFile: proto.String("foo.go"),
				Path:       []int32{4, 1, 2, 1},
				Begin:      proto.Int32(58),
				End:        proto.Int32(66),
				Semantic:   descriptorpb.GeneratedCodeInfo_Annotation_SET.Enum(),
			},
		},
	}

	content, err := g.Content()
	if err != nil {
		t.Fatalf("g.Content() = %v", err)
	}
	got, err := g.generatedCodeInfo(content)
	if err != nil {
		t.Fatalf("g.generatedCodeInfo(...) = %v", err)
	}
	if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
		t.Fatalf("GeneratedCodeInfo mismatch (-want +got):\n%s", diff)
	}
}
