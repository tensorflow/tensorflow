// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoregistry_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	"google.golang.org/protobuf/encoding/prototext"
	pimpl "google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	testpb "google.golang.org/protobuf/internal/testprotos/registry"
	"google.golang.org/protobuf/types/descriptorpb"
)

func mustMakeFile(s string) protoreflect.FileDescriptor {
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(pb, nil)
	if err != nil {
		panic(err)
	}
	return fd
}

func TestFiles(t *testing.T) {
	type (
		file struct {
			Path string
			Pkg  protoreflect.FullName
		}
		testFile struct {
			inFile  protoreflect.FileDescriptor
			wantErr string
		}
		testFindDesc struct {
			inName    protoreflect.FullName
			wantFound bool
		}
		testRangePkg struct {
			inPkg     protoreflect.FullName
			wantFiles []file
		}
		testFindPath struct {
			inPath    string
			wantFiles []file
			wantErr   string
		}
	)

	tests := []struct {
		files     []testFile
		findDescs []testFindDesc
		rangePkgs []testRangePkg
		findPaths []testFindPath
	}{{
		// Test that overlapping packages and files are permitted.
		files: []testFile{
			{inFile: mustMakeFile(`syntax:"proto2" name:"test1.proto" package:"foo.bar"`)},
			{inFile: mustMakeFile(`syntax:"proto2" name:"foo/bar/test.proto" package:"my.test"`)},
			{inFile: mustMakeFile(`syntax:"proto2" name:"foo/bar/test.proto" package:"foo.bar.baz"`), wantErr: "already registered"},
			{inFile: mustMakeFile(`syntax:"proto2" name:"test2.proto" package:"my.test.package"`)},
			{inFile: mustMakeFile(`syntax:"proto2" name:"weird" package:"foo.bar"`)},
			{inFile: mustMakeFile(`syntax:"proto2" name:"foo/bar/baz/../test.proto" package:"my.test"`)},
		},

		rangePkgs: []testRangePkg{{
			inPkg: "nothing",
		}, {
			inPkg: "",
		}, {
			inPkg: ".",
		}, {
			inPkg: "foo",
		}, {
			inPkg: "foo.",
		}, {
			inPkg: "foo..",
		}, {
			inPkg: "foo.bar",
			wantFiles: []file{
				{"test1.proto", "foo.bar"},
				{"weird", "foo.bar"},
			},
		}, {
			inPkg: "my.test",
			wantFiles: []file{
				{"foo/bar/baz/../test.proto", "my.test"},
				{"foo/bar/test.proto", "my.test"},
			},
		}, {
			inPkg: "fo",
		}},

		findPaths: []testFindPath{{
			inPath:  "nothing",
			wantErr: "not found",
		}, {
			inPath: "weird",
			wantFiles: []file{
				{"weird", "foo.bar"},
			},
		}, {
			inPath: "foo/bar/test.proto",
			wantFiles: []file{
				{"foo/bar/test.proto", "my.test"},
			},
		}},
	}, {
		// Test when new enum conflicts with existing package.
		files: []testFile{{
			inFile: mustMakeFile(`syntax:"proto2" name:"test1a.proto" package:"foo.bar.baz"`),
		}, {
			inFile:  mustMakeFile(`syntax:"proto2" name:"test1b.proto" enum_type:[{name:"foo" value:[{name:"VALUE" number:0}]}]`),
			wantErr: `file "test1b.proto" has a name conflict over foo`,
		}},
	}, {
		// Test when new package conflicts with existing enum.
		files: []testFile{{
			inFile: mustMakeFile(`syntax:"proto2" name:"test2a.proto" enum_type:[{name:"foo" value:[{name:"VALUE" number:0}]}]`),
		}, {
			inFile:  mustMakeFile(`syntax:"proto2" name:"test2b.proto" package:"foo.bar.baz"`),
			wantErr: `file "test2b.proto" has a package name conflict over foo`,
		}},
	}, {
		// Test when new enum conflicts with existing enum in same package.
		files: []testFile{{
			inFile: mustMakeFile(`syntax:"proto2" name:"test3a.proto" package:"foo" enum_type:[{name:"BAR" value:[{name:"VALUE" number:0}]}]`),
		}, {
			inFile:  mustMakeFile(`syntax:"proto2" name:"test3b.proto" package:"foo" enum_type:[{name:"BAR" value:[{name:"VALUE2" number:0}]}]`),
			wantErr: `file "test3b.proto" has a name conflict over foo.BAR`,
		}},
	}, {
		files: []testFile{{
			inFile: mustMakeFile(`
				syntax:  "proto2"
				name:    "test1.proto"
				package: "fizz.buzz"
				message_type: [{
					name: "Message"
					field: [
						{name:"Field" number:1 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:0}
					]
					oneof_decl:      [{name:"Oneof"}]
					extension_range: [{start:1000 end:2000}]

					enum_type: [
						{name:"Enum" value:[{name:"EnumValue" number:0}]}
					]
					nested_type: [
						{name:"Message" field:[{name:"Field" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]}
					]
					extension: [
						{name:"Extension" number:1001 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".fizz.buzz.Message"}
					]
				}]
				enum_type: [{
					name:  "Enum"
					value: [{name:"EnumValue" number:0}]
				}]
				extension: [
					{name:"Extension" number:1000 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".fizz.buzz.Message"}
				]
				service: [{
					name: "Service"
					method: [{
						name:             "Method"
						input_type:       ".fizz.buzz.Message"
						output_type:      ".fizz.buzz.Message"
						client_streaming: true
						server_streaming: true
					}]
				}]
			`),
		}, {
			inFile: mustMakeFile(`
				syntax:  "proto2"
				name:    "test2.proto"
				package: "fizz.buzz.gazz"
				enum_type: [{
					name:  "Enum"
					value: [{name:"EnumValue" number:0}]
				}]
			`),
		}, {
			inFile: mustMakeFile(`
				syntax:  "proto2"
				name:    "test3.proto"
				package: "fizz.buzz"
				enum_type: [{
					name:  "Enum1"
					value: [{name:"EnumValue1" number:0}]
				}, {
					name:  "Enum2"
					value: [{name:"EnumValue2" number:0}]
				}]
			`),
		}, {
			// Make sure we can register without package name.
			inFile: mustMakeFile(`
				name:   "weird"
				syntax: "proto2"
				message_type: [{
					name: "Message"
					nested_type: [{
						name: "Message"
						nested_type: [{
							name: "Message"
						}]
					}]
				}]
			`),
		}},
		findDescs: []testFindDesc{
			{inName: "fizz.buzz.message", wantFound: false},
			{inName: "fizz.buzz.Message", wantFound: true},
			{inName: "fizz.buzz.Message.X", wantFound: false},
			{inName: "fizz.buzz.Field", wantFound: false},
			{inName: "fizz.buzz.Oneof", wantFound: false},
			{inName: "fizz.buzz.Message.Field", wantFound: true},
			{inName: "fizz.buzz.Message.Field.X", wantFound: false},
			{inName: "fizz.buzz.Message.Oneof", wantFound: true},
			{inName: "fizz.buzz.Message.Oneof.X", wantFound: false},
			{inName: "fizz.buzz.Message.Message", wantFound: true},
			{inName: "fizz.buzz.Message.Message.X", wantFound: false},
			{inName: "fizz.buzz.Message.Enum", wantFound: true},
			{inName: "fizz.buzz.Message.Enum.X", wantFound: false},
			{inName: "fizz.buzz.Message.EnumValue", wantFound: true},
			{inName: "fizz.buzz.Message.EnumValue.X", wantFound: false},
			{inName: "fizz.buzz.Message.Extension", wantFound: true},
			{inName: "fizz.buzz.Message.Extension.X", wantFound: false},
			{inName: "fizz.buzz.enum", wantFound: false},
			{inName: "fizz.buzz.Enum", wantFound: true},
			{inName: "fizz.buzz.Enum.X", wantFound: false},
			{inName: "fizz.buzz.EnumValue", wantFound: true},
			{inName: "fizz.buzz.EnumValue.X", wantFound: false},
			{inName: "fizz.buzz.Enum.EnumValue", wantFound: false},
			{inName: "fizz.buzz.Extension", wantFound: true},
			{inName: "fizz.buzz.Extension.X", wantFound: false},
			{inName: "fizz.buzz.service", wantFound: false},
			{inName: "fizz.buzz.Service", wantFound: true},
			{inName: "fizz.buzz.Service.X", wantFound: false},
			{inName: "fizz.buzz.Method", wantFound: false},
			{inName: "fizz.buzz.Service.Method", wantFound: true},
			{inName: "fizz.buzz.Service.Method.X", wantFound: false},

			{inName: "fizz.buzz.gazz", wantFound: false},
			{inName: "fizz.buzz.gazz.Enum", wantFound: true},
			{inName: "fizz.buzz.gazz.EnumValue", wantFound: true},
			{inName: "fizz.buzz.gazz.Enum.EnumValue", wantFound: false},

			{inName: "fizz.buzz", wantFound: false},
			{inName: "fizz.buzz.Enum1", wantFound: true},
			{inName: "fizz.buzz.EnumValue1", wantFound: true},
			{inName: "fizz.buzz.Enum1.EnumValue1", wantFound: false},
			{inName: "fizz.buzz.Enum2", wantFound: true},
			{inName: "fizz.buzz.EnumValue2", wantFound: true},
			{inName: "fizz.buzz.Enum2.EnumValue2", wantFound: false},
			{inName: "fizz.buzz.Enum3", wantFound: false},

			{inName: "", wantFound: false},
			{inName: "Message", wantFound: true},
			{inName: "Message.Message", wantFound: true},
			{inName: "Message.Message.Message", wantFound: true},
			{inName: "Message.Message.Message.Message", wantFound: false},
		},
	}}

	sortFiles := cmpopts.SortSlices(func(x, y file) bool {
		return x.Path < y.Path || (x.Path == y.Path && x.Pkg < y.Pkg)
	})
	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var files protoregistry.Files
			for i, tc := range tt.files {
				gotErr := files.RegisterFile(tc.inFile)
				if ((gotErr == nil) != (tc.wantErr == "")) || !strings.Contains(fmt.Sprint(gotErr), tc.wantErr) {
					t.Errorf("file %d, Register() = %v, want %v", i, gotErr, tc.wantErr)
				}
			}

			for _, tc := range tt.findDescs {
				d, _ := files.FindDescriptorByName(tc.inName)
				gotFound := d != nil
				if gotFound != tc.wantFound {
					t.Errorf("FindDescriptorByName(%v) find mismatch: got %v, want %v", tc.inName, gotFound, tc.wantFound)
				}
			}

			for _, tc := range tt.rangePkgs {
				var gotFiles []file
				var gotCnt int
				wantCnt := files.NumFilesByPackage(tc.inPkg)
				files.RangeFilesByPackage(tc.inPkg, func(fd protoreflect.FileDescriptor) bool {
					gotFiles = append(gotFiles, file{fd.Path(), fd.Package()})
					gotCnt++
					return true
				})
				if gotCnt != wantCnt {
					t.Errorf("NumFilesByPackage(%v) = %v, want %v", tc.inPkg, gotCnt, wantCnt)
				}
				if diff := cmp.Diff(tc.wantFiles, gotFiles, sortFiles); diff != "" {
					t.Errorf("RangeFilesByPackage(%v) mismatch (-want +got):\n%v", tc.inPkg, diff)
				}
			}

			for _, tc := range tt.findPaths {
				var gotFiles []file
				fd, gotErr := files.FindFileByPath(tc.inPath)
				if gotErr == nil {
					gotFiles = append(gotFiles, file{fd.Path(), fd.Package()})
				}
				if ((gotErr == nil) != (tc.wantErr == "")) || !strings.Contains(fmt.Sprint(gotErr), tc.wantErr) {
					t.Errorf("FindFileByPath(%v) = %v, want %v", tc.inPath, gotErr, tc.wantErr)
				}
				if diff := cmp.Diff(tc.wantFiles, gotFiles, sortFiles); diff != "" {
					t.Errorf("FindFileByPath(%v) mismatch (-want +got):\n%v", tc.inPath, diff)
				}
			}
		})
	}
}

func TestTypes(t *testing.T) {
	mt1 := pimpl.Export{}.MessageTypeOf(&testpb.Message1{})
	et1 := pimpl.Export{}.EnumTypeOf(testpb.Enum1_ONE)
	xt1 := testpb.E_StringField
	xt2 := testpb.E_Message4_MessageField
	registry := new(protoregistry.Types)
	if err := registry.RegisterMessage(mt1); err != nil {
		t.Fatalf("registry.RegisterMessage(%v) returns unexpected error: %v", mt1.Descriptor().FullName(), err)
	}
	if err := registry.RegisterEnum(et1); err != nil {
		t.Fatalf("registry.RegisterEnum(%v) returns unexpected error: %v", et1.Descriptor().FullName(), err)
	}
	if err := registry.RegisterExtension(xt1); err != nil {
		t.Fatalf("registry.RegisterExtension(%v) returns unexpected error: %v", xt1.TypeDescriptor().FullName(), err)
	}
	if err := registry.RegisterExtension(xt2); err != nil {
		t.Fatalf("registry.RegisterExtension(%v) returns unexpected error: %v", xt2.TypeDescriptor().FullName(), err)
	}

	t.Run("FindMessageByName", func(t *testing.T) {
		tests := []struct {
			name         string
			messageType  protoreflect.MessageType
			wantErr      bool
			wantNotFound bool
		}{{
			name:        "testprotos.Message1",
			messageType: mt1,
		}, {
			name:         "testprotos.NoSuchMessage",
			wantErr:      true,
			wantNotFound: true,
		}, {
			name:    "testprotos.Enum1",
			wantErr: true,
		}, {
			name:    "testprotos.Enum2",
			wantErr: true,
		}, {
			name:    "testprotos.Enum3",
			wantErr: true,
		}}
		for _, tc := range tests {
			got, err := registry.FindMessageByName(protoreflect.FullName(tc.name))
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("FindMessageByName(%v) = (_, %v), want error? %t", tc.name, err, tc.wantErr)
				continue
			}
			if tc.wantNotFound && err != protoregistry.NotFound {
				t.Errorf("FindMessageByName(%v) got error: %v, want NotFound error", tc.name, err)
				continue
			}
			if got != tc.messageType {
				t.Errorf("FindMessageByName(%v) got wrong value: %v", tc.name, got)
			}
		}
	})

	t.Run("FindMessageByURL", func(t *testing.T) {
		tests := []struct {
			name         string
			messageType  protoreflect.MessageType
			wantErr      bool
			wantNotFound bool
		}{{
			name:        "testprotos.Message1",
			messageType: mt1,
		}, {
			name:         "type.googleapis.com/testprotos.Nada",
			wantErr:      true,
			wantNotFound: true,
		}, {
			name:    "testprotos.Enum1",
			wantErr: true,
		}}
		for _, tc := range tests {
			got, err := registry.FindMessageByURL(tc.name)
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("FindMessageByURL(%v) = (_, %v), want error? %t", tc.name, err, tc.wantErr)
				continue
			}
			if tc.wantNotFound && err != protoregistry.NotFound {
				t.Errorf("FindMessageByURL(%v) got error: %v, want NotFound error", tc.name, err)
				continue
			}
			if got != tc.messageType {
				t.Errorf("FindMessageByURL(%v) got wrong value: %v", tc.name, got)
			}
		}
	})

	t.Run("FindEnumByName", func(t *testing.T) {
		tests := []struct {
			name         string
			enumType     protoreflect.EnumType
			wantErr      bool
			wantNotFound bool
		}{{
			name:     "testprotos.Enum1",
			enumType: et1,
		}, {
			name:         "testprotos.None",
			wantErr:      true,
			wantNotFound: true,
		}, {
			name:    "testprotos.Message1",
			wantErr: true,
		}}
		for _, tc := range tests {
			got, err := registry.FindEnumByName(protoreflect.FullName(tc.name))
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("FindEnumByName(%v) = (_, %v), want error? %t", tc.name, err, tc.wantErr)
				continue
			}
			if tc.wantNotFound && err != protoregistry.NotFound {
				t.Errorf("FindEnumByName(%v) got error: %v, want NotFound error", tc.name, err)
				continue
			}
			if got != tc.enumType {
				t.Errorf("FindEnumByName(%v) got wrong value: %v", tc.name, got)
			}
		}
	})

	t.Run("FindExtensionByName", func(t *testing.T) {
		tests := []struct {
			name          string
			extensionType protoreflect.ExtensionType
			wantErr       bool
			wantNotFound  bool
		}{{
			name:          "testprotos.string_field",
			extensionType: xt1,
		}, {
			name:          "testprotos.Message4.message_field",
			extensionType: xt2,
		}, {
			name:         "testprotos.None",
			wantErr:      true,
			wantNotFound: true,
		}, {
			name:    "testprotos.Message1",
			wantErr: true,
		}}
		for _, tc := range tests {
			got, err := registry.FindExtensionByName(protoreflect.FullName(tc.name))
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("FindExtensionByName(%v) = (_, %v), want error? %t", tc.name, err, tc.wantErr)
				continue
			}
			if tc.wantNotFound && err != protoregistry.NotFound {
				t.Errorf("FindExtensionByName(%v) got error: %v, want NotFound error", tc.name, err)
				continue
			}
			if got != tc.extensionType {
				t.Errorf("FindExtensionByName(%v) got wrong value: %v", tc.name, got)
			}
		}
	})

	t.Run("FindExtensionByNumber", func(t *testing.T) {
		tests := []struct {
			parent        string
			number        int32
			extensionType protoreflect.ExtensionType
			wantErr       bool
			wantNotFound  bool
		}{{
			parent:        "testprotos.Message1",
			number:        11,
			extensionType: xt1,
		}, {
			parent:       "testprotos.Message1",
			number:       13,
			wantErr:      true,
			wantNotFound: true,
		}, {
			parent:        "testprotos.Message1",
			number:        21,
			extensionType: xt2,
		}, {
			parent:       "testprotos.Message1",
			number:       23,
			wantErr:      true,
			wantNotFound: true,
		}, {
			parent:       "testprotos.NoSuchMessage",
			number:       11,
			wantErr:      true,
			wantNotFound: true,
		}, {
			parent:       "testprotos.Message1",
			number:       30,
			wantErr:      true,
			wantNotFound: true,
		}, {
			parent:       "testprotos.Message1",
			number:       99,
			wantErr:      true,
			wantNotFound: true,
		}}
		for _, tc := range tests {
			got, err := registry.FindExtensionByNumber(protoreflect.FullName(tc.parent), protoreflect.FieldNumber(tc.number))
			gotErr := err != nil
			if gotErr != tc.wantErr {
				t.Errorf("FindExtensionByNumber(%v, %d) = (_, %v), want error? %t", tc.parent, tc.number, err, tc.wantErr)
				continue
			}
			if tc.wantNotFound && err != protoregistry.NotFound {
				t.Errorf("FindExtensionByNumber(%v, %d) got error %v, want NotFound error", tc.parent, tc.number, err)
				continue
			}
			if got != tc.extensionType {
				t.Errorf("FindExtensionByNumber(%v, %d) got wrong value: %v", tc.parent, tc.number, got)
			}
		}
	})

	sortTypes := cmp.Options{
		cmpopts.SortSlices(func(x, y protoreflect.EnumType) bool {
			return x.Descriptor().FullName() < y.Descriptor().FullName()
		}),
		cmpopts.SortSlices(func(x, y protoreflect.MessageType) bool {
			return x.Descriptor().FullName() < y.Descriptor().FullName()
		}),
		cmpopts.SortSlices(func(x, y protoreflect.ExtensionType) bool {
			return x.TypeDescriptor().FullName() < y.TypeDescriptor().FullName()
		}),
	}
	compare := cmp.Options{
		cmp.Comparer(func(x, y protoreflect.EnumType) bool {
			return x == y
		}),
		cmp.Comparer(func(x, y protoreflect.ExtensionType) bool {
			return x == y
		}),
		cmp.Comparer(func(x, y protoreflect.MessageType) bool {
			return x == y
		}),
	}

	t.Run("RangeEnums", func(t *testing.T) {
		want := []protoreflect.EnumType{et1}
		var got []protoreflect.EnumType
		var gotCnt int
		wantCnt := registry.NumEnums()
		registry.RangeEnums(func(et protoreflect.EnumType) bool {
			got = append(got, et)
			gotCnt++
			return true
		})

		if gotCnt != wantCnt {
			t.Errorf("NumEnums() = %v, want %v", gotCnt, wantCnt)
		}
		if diff := cmp.Diff(want, got, sortTypes, compare); diff != "" {
			t.Errorf("RangeEnums() mismatch (-want +got):\n%v", diff)
		}
	})

	t.Run("RangeMessages", func(t *testing.T) {
		want := []protoreflect.MessageType{mt1}
		var got []protoreflect.MessageType
		var gotCnt int
		wantCnt := registry.NumMessages()
		registry.RangeMessages(func(mt protoreflect.MessageType) bool {
			got = append(got, mt)
			gotCnt++
			return true
		})

		if gotCnt != wantCnt {
			t.Errorf("NumMessages() = %v, want %v", gotCnt, wantCnt)
		}
		if diff := cmp.Diff(want, got, sortTypes, compare); diff != "" {
			t.Errorf("RangeMessages() mismatch (-want +got):\n%v", diff)
		}
	})

	t.Run("RangeExtensions", func(t *testing.T) {
		want := []protoreflect.ExtensionType{xt1, xt2}
		var got []protoreflect.ExtensionType
		var gotCnt int
		wantCnt := registry.NumExtensions()
		registry.RangeExtensions(func(xt protoreflect.ExtensionType) bool {
			got = append(got, xt)
			gotCnt++
			return true
		})

		if gotCnt != wantCnt {
			t.Errorf("NumExtensions() = %v, want %v", gotCnt, wantCnt)
		}
		if diff := cmp.Diff(want, got, sortTypes, compare); diff != "" {
			t.Errorf("RangeExtensions() mismatch (-want +got):\n%v", diff)
		}
	})

	t.Run("RangeExtensionsByMessage", func(t *testing.T) {
		want := []protoreflect.ExtensionType{xt1, xt2}
		var got []protoreflect.ExtensionType
		var gotCnt int
		wantCnt := registry.NumExtensionsByMessage("testprotos.Message1")
		registry.RangeExtensionsByMessage("testprotos.Message1", func(xt protoreflect.ExtensionType) bool {
			got = append(got, xt)
			gotCnt++
			return true
		})

		if gotCnt != wantCnt {
			t.Errorf("NumExtensionsByMessage() = %v, want %v", gotCnt, wantCnt)
		}
		if diff := cmp.Diff(want, got, sortTypes, compare); diff != "" {
			t.Errorf("RangeExtensionsByMessage() mismatch (-want +got):\n%v", diff)
		}
	})
}
