// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodesc

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"

	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/types/descriptorpb"
)

func mustParseFile(s string) *descriptorpb.FileDescriptorProto {
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	return pb
}

func cloneFile(in *descriptorpb.FileDescriptorProto) *descriptorpb.FileDescriptorProto {
	return proto.Clone(in).(*descriptorpb.FileDescriptorProto)
}

var (
	proto2Enum = mustParseFile(`
		syntax:    "proto2"
		name:      "proto2_enum.proto"
		package:   "test.proto2"
		enum_type: [{name:"Enum" value:[{name:"ONE" number:1}]}]
	`)
	proto3Message = mustParseFile(`
		syntax:    "proto3"
		name:      "proto3_message.proto"
		package:   "test.proto3"
		message_type: [{
			name:  "Message"
			field: [
				{name:"foo" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
				{name:"bar" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}
			]
		}]
	`)
	protoEdition2023Message = mustParseFile(`
		syntax:    "editions"
		edition:   2023
		name:      "proto_editions_2023_message.proto"
		package:   "test.editions2023"
		options: {
			features: {
				field_presence: IMPLICIT
			}
		}
		message_type: [{
			name:  "Message"
			field: [
				{name:"foo" number:1 type:TYPE_STRING},
				{name:"bar" number:2 type:TYPE_STRING}
			]
		}]
	`)
	protoEdition2024Message = mustParseFile(`
		syntax:    "editions"
		edition:   2024
		name:      "proto_editions_2024_message.proto"
		package:   "test.editions2024"
		message_type: [{
			name:  "Message"
			field: [
				{
					name:"foo" number:1
					type:TYPE_STRING
				},
				{
					name:"bar" number:2
					type:TYPE_STRING
					options: {
						features: {
							field_presence: IMPLICIT
							utf8_validation: NONE
						}
					}
				}
			]
		}]
	`)
	extendableMessage = mustParseFile(`
		syntax:       "proto2"
		name:         "extendable_message.proto"
		package:      "test.proto2"
		message_type: [{name:"Message" extension_range:[{start:1 end:1000}]}]
	`)
	importPublicFile1 = mustParseFile(`
		syntax:            "proto3"
		name:              "import_public1.proto"
		dependency:        ["proto2_enum.proto", "proto3_message.proto", "extendable_message.proto"]
		message_type:      [{name:"Public1"}]
	`)
	importPublicFile2 = mustParseFile(`
		syntax:            "proto3"
		name:              "import_public2.proto"
		dependency:        ["import_public1.proto"]
		public_dependency: [0]
		message_type:      [{name:"Public2"}]
	`)
	importPublicFile3 = mustParseFile(`
		syntax:            "proto3"
		name:              "import_public3.proto"
		dependency:        ["import_public2.proto", "extendable_message.proto"]
		public_dependency: [0]
		message_type:      [{name:"Public3"}]
	`)
	importPublicFile4 = mustParseFile(`
		syntax:            "proto3"
		name:              "import_public4.proto"
		dependency:        ["import_public2.proto", "import_public3.proto", "proto2_enum.proto"]
		public_dependency: [0, 1]
		message_type:      [{name:"Public4"}]
	`)
)

func TestNewFile(t *testing.T) {
	tests := []struct {
		label    string
		inDeps   []*descriptorpb.FileDescriptorProto
		inDesc   *descriptorpb.FileDescriptorProto
		inOpts   FileOptions
		wantDesc *descriptorpb.FileDescriptorProto
		wantErr  string
	}{{
		label:   "empty path",
		inDesc:  mustParseFile(``),
		wantErr: `path must be populated`,
	}, {
		label:  "empty package and syntax",
		inDesc: mustParseFile(`name:"weird"`),
	}, {
		label:   "invalid syntax",
		inDesc:  mustParseFile(`name:"weird" syntax:"proto9"`),
		wantErr: `invalid syntax: "proto9"`,
	}, {
		label:   "bad package",
		inDesc:  mustParseFile(`name:"weird" package:"$"`),
		wantErr: `invalid package: "$"`,
	}, {
		label: "unresolvable import",
		inDesc: mustParseFile(`
			name:       "test.proto"
			dependency: "dep.proto"
		`),
		wantErr: `could not resolve import "dep.proto": not found`,
	}, {
		label: "unresolvable import but allowed",
		inDesc: mustParseFile(`
			name:       "test.proto"
			dependency: "dep.proto"
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
	}, {
		label: "duplicate import",
		inDesc: mustParseFile(`
			name:       "test.proto"
			dependency: ["dep.proto", "dep.proto"]
		`),
		inOpts:  FileOptions{AllowUnresolvable: true},
		wantErr: `already imported "dep.proto"`,
	}, {
		label: "invalid weak import",
		inDesc: mustParseFile(`
			name:            "test.proto"
			dependency:      "dep.proto"
			weak_dependency: [-23]
		`),
		inOpts:  FileOptions{AllowUnresolvable: true},
		wantErr: `invalid or duplicate weak import index: -23`,
	}, {
		label: "normal weak and public import",
		inDesc: mustParseFile(`
			name:              "test.proto"
			dependency:        "dep.proto"
			weak_dependency:   [0]
			public_dependency: [0]
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
	}, {
		label: "import public indirect dependency duplicate",
		inDeps: []*descriptorpb.FileDescriptorProto{
			mustParseFile(`name:"leaf.proto"`),
			mustParseFile(`name:"public.proto" dependency:"leaf.proto" public_dependency:0`),
		},
		inDesc: mustParseFile(`
			name: "test.proto"
			dependency: ["public.proto", "leaf.proto"]
		`),
	}, {
		label: "import public graph",
		inDeps: []*descriptorpb.FileDescriptorProto{
			cloneFile(proto2Enum),
			cloneFile(proto3Message),
			cloneFile(extendableMessage),
			cloneFile(importPublicFile1),
			cloneFile(importPublicFile2),
			cloneFile(importPublicFile3),
			cloneFile(importPublicFile4),
		},
		inDesc: mustParseFile(`
			name:       "test.proto"
			package:    "test.graph"
			dependency: ["import_public4.proto"],
		`),
		// TODO: Test import public
	}, {
		label: "preserve source code locations",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			source_code_info: {location: [{
				span: [39,0,882,1]
			}, {
				path: [12]
				span: [39,0,18]
				leading_detached_comments: [" foo\n"," bar\n"]
			}, {
				path: [8,9]
				span: [51,0,28]
				leading_comments: " Comment\n"
			}]}
		`),
	}, {
		label: "invalid source code span",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			source_code_info: {location: [{
				span: [39]
			}]}
		`),
		wantErr: `invalid span: [39]`,
	}, {
		label: "resolve relative reference",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "A"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:"B.C"}]
				nested_type: [{name: "B"}]
			}, {
				name: "B"
				nested_type: [{name: "C"}]
			}]
		`),
		wantDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "A"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".fizz.buzz.B.C"}]
				nested_type: [{name: "B"}]
			}, {
				name: "B"
				nested_type: [{name: "C"}]
			}]
		`),
	}, {
		label: "resolve the wrong type",
		inDesc: mustParseFile(`
			name: "test.proto"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:"E"}]
				enum_type: [{name: "E" value: [{name:"V0" number:0}, {name:"V1" number:1}]}]
			}]
		`),
		wantErr: `message field "M.F" cannot resolve type: resolved "M.E", but it is not an message`,
	}, {
		label: "auto-resolve unknown kind",
		inDesc: mustParseFile(`
			name: "test.proto"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type_name:"E"}]
				enum_type: [{name: "E" value: [{name:"V0" number:0}, {name:"V1" number:1}]}]
			}]
		`),
		wantDesc: mustParseFile(`
			name: "test.proto"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:".M.E"}]
				enum_type: [{name: "E" value: [{name:"V0" number:0}, {name:"V1" number:1}]}]
			}]
		`),
	}, {
		label: "unresolved import",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			dependency: "remote.proto"
		`),
		wantErr: `could not resolve import "remote.proto": not found`,
	}, {
		label: "unresolved message field",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "M"
				field: [{name:"F1" number:1 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:"some.other.enum" default_value:"UNKNOWN"}]
			}]
		`),
		wantErr: `message field "fizz.buzz.M.F1" cannot resolve type: "*.some.other.enum" not found`,
	}, {
		label: "unresolved default enum value",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "M"
				field: [{name:"F1" number:1 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:"E" default_value:"UNKNOWN"}]
				enum_type: [{name:"E" value:[{name:"V0" number:0}]}]
			}]
		`),
		wantErr: `message field "fizz.buzz.M.F1" has invalid default: could not parse value for enum: "UNKNOWN"`,
	}, {
		label: "allowed unresolved default enum value",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "M"
				field: [{name:"F1" number:1 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:".fizz.buzz.M.E" default_value:"UNKNOWN"}]
				enum_type: [{name:"E" value:[{name:"V0" number:0}]}]
			}]
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
	}, {
		label: "unresolved extendee",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			extension: [{name:"X" number:1 label:LABEL_OPTIONAL extendee:"some.extended.message" type:TYPE_MESSAGE type_name:"some.other.message"}]
		`),
		wantErr: `extension field "fizz.buzz.X" cannot resolve extendee: "*.some.extended.message" not found`,
	}, {
		label: "unresolved method input",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			service: [{
				name: "S"
				method: [{name:"M" input_type:"foo.bar.input" output_type:".absolute.foo.bar.output"}]
			}]
		`),
		wantErr: `service method "fizz.buzz.S.M" cannot resolve input: "*.foo.bar.input" not found`,
	}, {
		label: "allowed unresolved references",
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			dependency: "remote.proto"
			message_type: [{
				name: "M"
				field: [{name:"F1" number:1 label:LABEL_OPTIONAL type_name:"some.other.enum" default_value:"UNKNOWN"}]
			}]
			extension: [{name:"X" number:1 label:LABEL_OPTIONAL extendee:"some.extended.message" type:TYPE_MESSAGE type_name:"some.other.message"}]
			service: [{
				name: "S"
				method: [{name:"M" input_type:"foo.bar.input" output_type:".absolute.foo.bar.output"}]
			}]
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
	}, {
		label: "resolved but not imported",
		inDeps: []*descriptorpb.FileDescriptorProto{mustParseFile(`
			name: "dep.proto"
			package: "fizz"
			message_type: [{name:"M" nested_type:[{name:"M"}]}]
		`)},
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:"M.M"}]
			}]
		`),
		wantErr: `message field "fizz.buzz.M.F" cannot resolve type: resolved "fizz.M.M", but "dep.proto" is not imported`,
	}, {
		label: "resolved from remote import",
		inDeps: []*descriptorpb.FileDescriptorProto{mustParseFile(`
			name: "dep.proto"
			package: "fizz"
			message_type: [{name:"M" nested_type:[{name:"M"}]}]
		`)},
		inDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			dependency: "dep.proto"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:"M.M"}]
			}]
		`),
		wantDesc: mustParseFile(`
			name: "test.proto"
			package: "fizz.buzz"
			dependency: "dep.proto"
			message_type: [{
				name: "M"
				field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".fizz.M.M"}]
			}]
		`),
	}, {
		label: "namespace conflict on enum value",
		inDesc: mustParseFile(`
			name:    "test.proto"
			enum_type: [{
				name: "foo"
				value: [{name:"foo" number:0}]
			}]
		`),
		wantErr: `descriptor "foo" already declared`,
	}, {
		label: "no namespace conflict on message field",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{
				name: "foo"
				field: [{name:"foo" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]
		`),
	}, {
		label: "invalid name",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name: "$"}]
		`),
		wantErr: `descriptor "" has an invalid nested name: "$"`,
	}, {
		label: "invalid empty enum",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{name:"E"}]}]
		`),
		wantErr: `enum "M.E" must contain at least one value declaration`,
	}, {
		label: "invalid enum value without number",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{name:"E" value:[{name:"one"}]}]}]
		`),
		wantErr: `enum value "M.one" must have a specified number`,
	}, {
		label: "valid enum",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{name:"E" value:[{name:"one" number:1}]}]}]
		`),
	}, {
		label: "invalid enum reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:          "E"
				reserved_name: [""]
				value: [{name:"V" number:0}]
			}]}]
		`),
		// NOTE: In theory this should be an error.
		// See https://github.com/protocolbuffers/protobuf/issues/6335.
		/*wantErr: `enum "M.E" reserved names has invalid name: ""`,*/
	}, {
		label: "duplicate enum reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:          "E"
				reserved_name: ["foo", "foo"]
			}]}]
		`),
		wantErr: `enum "M.E" reserved names has duplicate name: "foo"`,
	}, {
		label: "valid enum reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:          "E"
				reserved_name: ["foo", "bar"]
				value:         [{name:"baz" number:1}]
			}]}]
		`),
	}, {
		label: "use of enum reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:          "E"
				reserved_name: ["foo", "bar"]
				value:         [{name:"foo" number:1}]
			}]}]
		`),
		wantErr: `enum value "M.foo" must not use reserved name`,
	}, {
		label: "invalid enum reserved ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:           "E"
				reserved_range: [{start:5 end:4}]
			}]}]
		`),
		wantErr: `enum "M.E" reserved ranges has invalid range: 5 to 4`,
	}, {
		label: "overlapping enum reserved ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:           "E"
				reserved_range: [{start:1 end:1000}, {start:10 end:100}]
			}]}]
		`),
		wantErr: `enum "M.E" reserved ranges has overlapping ranges: 1 to 1000 with 10 to 100`,
	}, {
		label: "valid enum reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:           "E"
				reserved_range: [{start:1 end:10}, {start:100 end:1000}]
				value:          [{name:"baz" number:50}]
			}]}]
		`),
	}, {
		label: "use of enum reserved range",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:           "E"
				reserved_range: [{start:1 end:10}, {start:100 end:1000}]
				value:          [{name:"baz" number:500}]
			}]}]
		`),
		wantErr: `enum value "M.baz" must not use reserved number 500`,
	}, {
		label: "unused enum alias feature",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:    "E"
				value:   [{name:"baz" number:500}]
				options: {allow_alias:true}
			}]}]
		`),
		wantErr: `enum "M.E" allows aliases, but none were found`,
	}, {
		label: "enum number conflicts",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"foo" number:0}, {name:"bar" number:1}, {name:"baz" number:1}]
			}]}]
		`),
		wantErr: `enum "M.E" has conflicting non-aliased values on number 1: "baz" with "bar"`,
	}, {
		label: "aliased enum numbers",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:    "E"
				value:   [{name:"foo" number:0}, {name:"bar" number:1}, {name:"baz" number:1}]
				options: {allow_alias:true}
			}]}]
		`),
	}, {
		label: "invalid proto3 enum",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"baz" number:500}]
			}]}]
		`),
		wantErr: `enum "M.baz" using proto3 semantics must have zero number for the first value`,
	}, {
		label: "valid proto3 enum",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"baz" number:0}]
			}]}]
		`),
	}, {
		label: "proto3 enum name prefix conflict",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"e_Foo" number:0}, {name:"fOo" number:1}]
			}]}]
		`),
		wantErr: `enum "M.E" using proto3 semantics has conflict: "fOo" with "e_Foo"`,
	}, {
		label: "proto2 enum has name prefix check",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"e_Foo" number:0}, {name:"fOo" number:1}]
			}]}]
		`),
	}, {
		label: "proto3 enum same name prefix with number conflict",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:  "E"
				value: [{name:"e_Foo" number:0}, {name:"fOo" number:0}]
			}]}]
		`),
		wantErr: `enum "M.E" has conflicting non-aliased values on number 0: "fOo" with "e_Foo"`,
	}, {
		label: "proto3 enum same name prefix with alias numbers",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" enum_type:[{
				name:    "E"
				value:   [{name:"e_Foo" number:0}, {name:"fOo" number:0}]
				options: {allow_alias: true}
			}]}]
		`),
	}, {
		label: "invalid message reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:          "M"
				reserved_name: ["$"]
			}]}]
		`),
		// NOTE: In theory this should be an error.
		// See https://github.com/protocolbuffers/protobuf/issues/6335.
		/*wantErr: `message "M.M" reserved names has invalid name: "$"`,*/
	}, {
		label: "valid message reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:          "M"
				reserved_name: ["foo", "bar"]
				field:         [{name:"foo" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message field "M.M.foo" must not use reserved name`,
	}, {
		label: "valid message reserved names",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:          "M"
				reserved_name: ["foo", "bar"]
				field:         [{name:"baz" number:1 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:0}]
				oneof_decl:    [{name:"foo"}] # not affected by reserved_name
			}]}]
		`),
	}, {
		label: "invalid reserved number",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:           "M"
				reserved_range: [{start:1 end:1}]
				field:          [{name:"baz" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message "M.M" reserved ranges has invalid field number: 0`,
	}, {
		label: "invalid reserved ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:           "M"
				reserved_range: [{start:2 end:2}]
				field:          [{name:"baz" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message "M.M" reserved ranges has invalid range: 2 to 1`,
	}, {
		label: "overlapping reserved ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:           "M"
				reserved_range: [{start:1 end:10}, {start:2 end:9}]
				field:          [{name:"baz" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message "M.M" reserved ranges has overlapping ranges: 1 to 9 with 2 to 8`,
	}, {
		label: "use of reserved message field number",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:           "M"
				reserved_range: [{start:10 end:20}, {start:20 end:30}, {start:30 end:31}]
				field:          [{name:"baz" number:30 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message field "M.M.baz" must not use reserved number 30`,
	}, {
		label: "invalid extension ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:            "M"
				extension_range: [{start:-500 end:2}]
				field:           [{name:"baz" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}]
			}]}]
		`),
		wantErr: `message "M.M" extension ranges has invalid field number: -500`,
	}, {
		label: "overlapping reserved and extension ranges",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:            "M"
				reserved_range:  [{start:15 end:20}, {start:1 end:3}, {start:7 end:10}]
				extension_range: [{start:8 end:9}, {start:3 end:5}]
			}]}]
		`),
		wantErr: `message "M.M" reserved and extension ranges has overlapping ranges: 7 to 9 with 8`,
	}, {
		label: "message field conflicting number",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:            "M"
				field: [
					{name:"one" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
					{name:"One" number:1 label:LABEL_OPTIONAL type:TYPE_STRING}
				]
			}]}]
		`),
		wantErr: `message "M.M" has conflicting fields: "One" with "one"`,
	}, {
		label: "invalid MessageSet",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:    "M"
				options: {message_set_wire_format:true}
			}]}]
		`),
		wantErr: func() string {
			if flags.ProtoLegacy {
				return `message "M.M" is an invalid proto1 MessageSet`
			} else {
				return `message "M.M" is a MessageSet, which is a legacy proto1 feature that is no longer supported`
			}
		}(),
	}, {
		label: "valid MessageSet",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:            "M"
				extension_range: [{start:1 end:100000}]
				options:         {message_set_wire_format:true}
			}]}]
		`),
		wantErr: func() string {
			if flags.ProtoLegacy {
				return ""
			} else {
				return `message "M.M" is a MessageSet, which is a legacy proto1 feature that is no longer supported`
			}
		}(),
	}, {
		label: "invalid extension ranges in proto3",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:            "M"
				extension_range: [{start:1 end:100000}]
			}]}]
		`),
		wantErr: `message "M.M" using proto3 semantics cannot have extension ranges`,
	}, {
		label: "proto3 message fields conflict",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name: "M"
				field: [
					{name:"_b_a_z_" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
					{name:"baz" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}
				]
			}]}]
		`),
		wantErr: `message "M.M" using proto3 semantics has conflict: "baz" with "_b_a_z_"`,
	}, {
		label: "proto3 message fields",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:       "M"
				field:      [{name:"_b_a_z_" number:1 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:0}]
				oneof_decl: [{name:"baz"}] # proto3 name conflict logic does not include oneof
			}]}]
		`),
	}, {
		label: "proto3 message field with defaults",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:       "M"
				field:      [{name:"a" number:1 type:TYPE_STRING default_value:"abc"}]
			}]}]
		`),
		wantErr: `message field "M.M.a" has invalid default: cannot be specified with implicit field presence`,
	}, {
		label: "proto editions implicit presence field with defaults",
		inDesc: mustParseFile(`
			syntax:  "proto3"
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name:       "M"
				field:      [{name:"a" number:1 type:TYPE_STRING default_value:"abc" options:{features:{field_presence:IMPLICIT}}}]
			}]}]
		`),
		wantErr: `message field "M.M.a" has invalid default: cannot be specified with implicit field presence`,
	}, {
		label: "proto2 message fields with no conflict",
		inDesc: mustParseFile(`
			name:    "test.proto"
			message_type: [{name:"M" nested_type:[{
				name: "M"
				field: [
					{name:"_b_a_z_" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
					{name:"baz" number:2 label:LABEL_OPTIONAL type:TYPE_STRING}
				]
			}]}]
		`),
	}, {
		label: "proto3 message with unresolved enum",
		inDesc: mustParseFile(`
			name:    "test.proto"
			syntax:  "proto3"
			message_type: [{
				name: "M"
				field: [
					{name:"enum" number:1 label:LABEL_OPTIONAL type:TYPE_ENUM type_name:".fizz.buzz.Enum"}
				]
			}]
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
		// TODO: Test field and oneof handling in validateMessageDeclarations
		// TODO: Test unmarshalDefault
		// TODO: Test validateExtensionDeclarations
		// TODO: Test checkValidGroup
		// TODO: Test checkValidMap
	}, {
		label: "empty service",
		inDesc: mustParseFile(`
			name:    "test.proto"
			service: [{name:"service"}]
		`),
	}, {
		label: "service with method with unresolved",
		inDesc: mustParseFile(`
			name:    "test.proto"
			service: [{
				name: "service"
				method: [{
					name:"method"
					input_type:"foo"
					output_type:".foo.bar.baz"
				}]
			}]
		`),
		inOpts: FileOptions{AllowUnresolvable: true},
	}, {
		label: "service with wrong reference type",
		inDeps: []*descriptorpb.FileDescriptorProto{
			cloneFile(proto3Message),
			cloneFile(proto2Enum),
		},
		inDesc: mustParseFile(`
			name:    "test.proto"
			dependency: ["proto2_enum.proto", "proto3_message.proto"]
			service: [{
				name: "service"
				method: [{
					name:        "method"
					input_type:  ".test.proto2.Enum",
					output_type: ".test.proto3.Message"
				}]
			}]
		`),
		wantErr: `service method "service.method" cannot resolve input: resolved "test.proto2.Enum", but it is not an message`,
	}}

	for _, tt := range tests {
		t.Run(tt.label, func(t *testing.T) {
			r := new(protoregistry.Files)
			for i, dep := range tt.inDeps {
				f, err := tt.inOpts.New(dep, r)
				if err != nil {
					t.Fatalf("dependency %d: unexpected NewFile() error: %v", i, err)
				}
				if err := r.RegisterFile(f); err != nil {
					t.Fatalf("dependency %d: unexpected Register() error: %v", i, err)
				}
			}
			var gotDesc *descriptorpb.FileDescriptorProto
			if tt.wantErr == "" && tt.wantDesc == nil {
				tt.wantDesc = cloneFile(tt.inDesc)
			}
			gotFile, err := tt.inOpts.New(tt.inDesc, r)
			if gotFile != nil {
				gotDesc = ToFileDescriptorProto(gotFile)
			}
			if !proto.Equal(gotDesc, tt.wantDesc) {
				t.Errorf("NewFile() mismatch:\ngot  %v\nwant %v", gotDesc, tt.wantDesc)
			}
			if ((err == nil) != (tt.wantErr == "")) || !strings.Contains(fmt.Sprint(err), tt.wantErr) {
				t.Errorf("NewFile() error:\ngot:  %v\nwant: %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewFiles(t *testing.T) {
	fdset := &descriptorpb.FileDescriptorSet{
		File: []*descriptorpb.FileDescriptorProto{
			mustParseFile(`
				name: "test.proto"
				package: "fizz"
				dependency: "dep.proto"
				message_type: [{
					name: "M2"
					field: [{name:"F" number:1 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:"M1"}]
				}]
			`),
			// Inputs deliberately out of order.
			mustParseFile(`
				name: "dep.proto"
				package: "fizz"
				message_type: [{name:"M1"}]
			`),
		},
	}
	f, err := NewFiles(fdset)
	if err != nil {
		t.Fatal(err)
	}
	m1, err := f.FindDescriptorByName("fizz.M1")
	if err != nil {
		t.Fatalf(`f.FindDescriptorByName("fizz.M1") = %v`, err)
	}
	m2, err := f.FindDescriptorByName("fizz.M2")
	if err != nil {
		t.Fatalf(`f.FindDescriptorByName("fizz.M2") = %v`, err)
	}
	if m2.(protoreflect.MessageDescriptor).Fields().ByName("F").Message() != m1 {
		t.Fatalf(`m1.Fields().ByName("F").Message() != m2`)
	}
}

func TestNewFilesImportCycle(t *testing.T) {
	fdset := &descriptorpb.FileDescriptorSet{
		File: []*descriptorpb.FileDescriptorProto{
			mustParseFile(`
				name: "test.proto"
				package: "fizz"
				dependency: "dep.proto"
			`),
			mustParseFile(`
				name: "dep.proto"
				package: "fizz"
				dependency: "test.proto"
			`),
		},
	}
	_, err := NewFiles(fdset)
	if err == nil {
		t.Fatal("NewFiles with import cycle: success, want error")
	}
}

func TestSourceLocations(t *testing.T) {
	fd := mustParseFile(`
		name: "comments.proto"
		message_type: [{
			name: "Message1"
			field: [
				{name:"field1" number:1 label:LABEL_OPTIONAL type:TYPE_STRING},
				{name:"field2" number:2 label:LABEL_OPTIONAL type:TYPE_STRING},
				{name:"field3" number:3 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:0},
				{name:"field4" number:4 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:0},
				{name:"field5" number:5 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:1},
				{name:"field6" number:6 label:LABEL_OPTIONAL type:TYPE_STRING oneof_index:1}
			]
			extension: [
				{name:"extension1" number:100 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".Message1"},
				{name:"extension2" number:101 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".Message1"}
			]
			nested_type: [{name:"Message1"}, {name:"Message2"}]
			extension_range: {start:100 end:536870912}
			oneof_decl: [{name:"oneof1"}, {name:"oneof2"}]
		}, {
			name: "Message2"
			enum_type: {
				name: "Enum1"
				value: [
					{name: "FOO", number: 0},
					{name: "BAR", number: 1}
				]
			}
		}]
		enum_type: {
			name: "Enum1"
			value: [
				{name: "FOO", number: 0},
				{name: "BAR", number: 1}
			]
		}
		service: {
			name: "Service1"
			method: [
				{name:"Method1" input_type:".Message1" output_type:".Message1"},
				{name:"Method2" input_type:".Message2" output_type:".Message2"}
			]
		}
		extension: [
			{name:"extension1" number:102 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".Message1"},
			{name:"extension2" number:103 label:LABEL_OPTIONAL type:TYPE_STRING extendee:".Message1"}
		]
		source_code_info: {
			location: [
				{span:[0,0,69,1]},
				{path:[12] span:[0,0,18]},
				{path:[5,0] span:[3,0,8,1] leading_comments:" Enum1\r\n"},
				{path:[5,0,1] span:[3,5,10]},
				{path:[5,0,2,0] span:[5,2,10] leading_comments:" FOO\r\n"},
				{path:[5,0,2,0,1] span:[5,2,5]},
				{path:[5,0,2,0,2] span:[5,8,9]},
				{path:[5,0,2,1] span:[7,2,10] leading_comments:" BAR\r\n"},
				{path:[5,0,2,1,1] span:[7,2,5]},
				{path:[5,0,2,1,2] span:[7,8,9]},
				{path:[4,0] span:[11,0,43,1] leading_comments:" Message1\r\n"},
				{path:[4,0,1] span:[11,8,16]},
				{path:[4,0,3,0] span:[13,2,21] leading_comments:" Message1.Message1\r\n"},
				{path:[4,0,3,0,1] span:[13,10,18]},
				{path:[4,0,3,1] span:[15,2,21] leading_comments:" Message1.Message2\r\n"},
				{path:[4,0,3,1,1] span:[15,10,18]},
				{path:[4,0,2,0] span:[18,2,29] leading_comments:" Message1.field1\r\n"},
				{path:[4,0,2,0,4] span:[18,2,10]},
				{path:[4,0,2,0,5] span:[18,11,17]},
				{path:[4,0,2,0,1] span:[18,18,24]},
				{path:[4,0,2,0,3] span:[18,27,28]},
				{path:[4,0,2,1] span:[20,2,29] leading_comments:" Message1.field2\r\n"},
				{path:[4,0,2,1,4] span:[20,2,10]},
				{path:[4,0,2,1,5] span:[20,11,17]},
				{path:[4,0,2,1,1] span:[20,18,24]},
				{path:[4,0,2,1,3] span:[20,27,28]},
				{path:[4,0,8,0] span:[22,2,27,3] leading_comments:" Message1.oneof1\r\n"},
				{path:[4,0,8,0,1] span:[22,8,14]},
				{path:[4,0,2,2] span:[24,4,22] leading_comments:" Message1.field3\r\n"},
				{path:[4,0,2,2,5] span:[24,4,10]},
				{path:[4,0,2,2,1] span:[24,11,17]},
				{path:[4,0,2,2,3] span:[24,20,21]},
				{path:[4,0,2,3] span:[26,4,22] leading_comments:" Message1.field4\r\n"},
				{path:[4,0,2,3,5] span:[26,4,10]},
				{path:[4,0,2,3,1] span:[26,11,17]},
				{path:[4,0,2,3,3] span:[26,20,21]},
				{path:[4,0,8,1] span:[29,2,34,3] leading_comments:" Message1.oneof2\r\n"},
				{path:[4,0,8,1,1] span:[29,8,14]},
				{path:[4,0,2,4] span:[31,4,22] leading_comments:" Message1.field5\r\n"},
				{path:[4,0,2,4,5] span:[31,4,10]},
				{path:[4,0,2,4,1] span:[31,11,17]},
				{path:[4,0,2,4,3] span:[31,20,21]},
				{path:[4,0,2,5] span:[33,4,22] leading_comments:" Message1.field6\r\n"},
				{path:[4,0,2,5,5] span:[33,4,10]},
				{path:[4,0,2,5,1] span:[33,11,17]},
				{path:[4,0,2,5,3] span:[33,20,21]},
				{path:[4,0,5] span:[36,2,24]},
				{path:[4,0,5,0] span:[36,13,23]},
				{path:[4,0,5,0,1] span:[36,13,16]},
				{path:[4,0,5,0,2] span:[36,20,23]},
				{path:[4,0,6] span:[37,2,42,3]},
				{path:[4,0,6,0] span:[39,4,37] leading_comments:" Message1.extension1\r\n"},
				{path:[4,0,6,0,2] span:[37,9,18]},
				{path:[4,0,6,0,4] span:[39,4,12]},
				{path:[4,0,6,0,5] span:[39,13,19]},
				{path:[4,0,6,0,1] span:[39,20,30]},
				{path:[4,0,6,0,3] span:[39,33,36]},
				{path:[4,0,6,1] span:[41,4,37] leading_comments:" Message1.extension2\r\n"},
				{path:[4,0,6,1,2] span:[37,9,18]},
				{path:[4,0,6,1,4] span:[41,4,12]},
				{path:[4,0,6,1,5] span:[41,13,19]},
				{path:[4,0,6,1,1] span:[41,20,30]},
				{path:[4,0,6,1,3] span:[41,33,36]},
				{path:[7] span:[45,0,50,1]},
				{path:[7,0] span:[47,2,35] leading_comments:" extension1\r\n"},
				{path:[7,0,2] span:[45,7,15]},
				{path:[7,0,4] span:[47,2,10]},
				{path:[7,0,5] span:[47,11,17]},
				{path:[7,0,1] span:[47,18,28]},
				{path:[7,0,3] span:[47,31,34]},
				{path:[7,1] span:[49,2,35] leading_comments:" extension2\r\n"},
				{path:[7,1,2] span:[45,7,15]},
				{path:[7,1,4] span:[49,2,10]},
				{path:[7,1,5] span:[49,11,17]},
				{path:[7,1,1] span:[49,18,28]},
				{path:[7,1,3] span:[49,31,34]},
				{path:[4,1] span:[53,0,61,1] leading_comments:" Message2\r\n"},
				{path:[4,1,1] span:[53,8,16]},
				{path:[4,1,4,0] span:[55,2,60,3] leading_comments:" Message2.Enum1\r\n"},
				{path:[4,1,4,0,1] span:[55,7,12]},
				{path:[4,1,4,0,2,0] span:[57,4,12] leading_comments:" Message2.FOO\r\n"},
				{path:[4,1,4,0,2,0,1] span:[57,4,7]},
				{path:[4,1,4,0,2,0,2] span:[57,10,11]},
				{path:[4,1,4,0,2,1] span:[59,4,12] leading_comments:" Message2.BAR\r\n"},
				{path:[4,1,4,0,2,1,1] span:[59,4,7]},
				{path:[4,1,4,0,2,1,2] span:[59,10,11]},
				{path:[6,0] span:[64,0,69,1] leading_comments:" Service1\r\n"},
				{path:[6,0,1] span:[64,8,16]},
				{path:[6,0,2,0] span:[66,2,43] leading_comments:" Service1.Method1\r\n"},
				{path:[6,0,2,0,1] span:[66,6,13]},
				{path:[6,0,2,0,2] span:[66,14,22]},
				{path:[6,0,2,0,3] span:[66,33,41]},
				{path:[6,0,2,1] span:[68,2,43] leading_comments:" Service1.Method2\r\n"},
				{path:[6,0,2,1,1] span:[68,6,13]},
				{path:[6,0,2,1,2] span:[68,14,22]},
				{path:[6,0,2,1,3] span:[68,33,41]}
			]
		}
	`)
	fileDesc, err := NewFile(fd, nil)
	if err != nil {
		t.Fatalf("NewFile error: %v", err)
	}

	var walkDescs func(protoreflect.Descriptor, func(protoreflect.Descriptor))
	walkDescs = func(d protoreflect.Descriptor, f func(protoreflect.Descriptor)) {
		f(d)
		if d, ok := d.(interface {
			Enums() protoreflect.EnumDescriptors
		}); ok {
			eds := d.Enums()
			for i := 0; i < eds.Len(); i++ {
				walkDescs(eds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Values() protoreflect.EnumValueDescriptors
		}); ok {
			vds := d.Values()
			for i := 0; i < vds.Len(); i++ {
				walkDescs(vds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Messages() protoreflect.MessageDescriptors
		}); ok {
			mds := d.Messages()
			for i := 0; i < mds.Len(); i++ {
				walkDescs(mds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Fields() protoreflect.FieldDescriptors
		}); ok {
			fds := d.Fields()
			for i := 0; i < fds.Len(); i++ {
				walkDescs(fds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Oneofs() protoreflect.OneofDescriptors
		}); ok {
			ods := d.Oneofs()
			for i := 0; i < ods.Len(); i++ {
				walkDescs(ods.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Extensions() protoreflect.ExtensionDescriptors
		}); ok {
			xds := d.Extensions()
			for i := 0; i < xds.Len(); i++ {
				walkDescs(xds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Services() protoreflect.ServiceDescriptors
		}); ok {
			sds := d.Services()
			for i := 0; i < sds.Len(); i++ {
				walkDescs(sds.Get(i), f)
			}
		}
		if d, ok := d.(interface {
			Methods() protoreflect.MethodDescriptors
		}); ok {
			mds := d.Methods()
			for i := 0; i < mds.Len(); i++ {
				walkDescs(mds.Get(i), f)
			}
		}
	}

	var numDescs int
	walkDescs(fileDesc, func(d protoreflect.Descriptor) {
		// The comment for every descriptor should be the full name itself.
		got := strings.TrimSpace(fileDesc.SourceLocations().ByDescriptor(d).LeadingComments)
		want := string(d.FullName())
		if got != want {
			t.Errorf("comment mismatch: got %v, want %v", got, want)
		}
		numDescs++
	})
	if numDescs != 30 {
		t.Errorf("visited %d descriptor, expected 30", numDescs)
	}
}

func TestToFileDescriptorProtoPlaceHolder(t *testing.T) {
	// Make sure placeholders produce valid protos.
	fileDescriptor := ToFileDescriptorProto(filedesc.PlaceholderFile("foo/test.proto"))
	_, err := NewFile(fileDescriptor, &protoregistry.Files{} /* empty files since placeholder has no deps */)
	if err != nil {
		t.Errorf("placeholder file descriptor proto is not valid: %s", err)
	}
}
