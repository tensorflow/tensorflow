// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"google.golang.org/protobuf/types/descriptorpb"
)

func mustMakeMessage(s string) *descriptorpb.DescriptorProto {
	s = fmt.Sprintf(`name:"test.proto" syntax:"proto2" message_type:[{%s}]`, s)
	pb := new(descriptorpb.FileDescriptorProto)
	if err := prototext.Unmarshal([]byte(s), pb); err != nil {
		panic(err)
	}
	return pb.MessageType[0]
}

func TestFields(t *testing.T) {
	type fieldsKind struct {
		kind   protoreflect.Kind
		fields string
	}
	tests := []struct {
		inFields []fieldsKind
		wantMsg  *descriptorpb.DescriptorProto
		wantErr  string
	}{{
		inFields: []fieldsKind{{protoreflect.MessageKind, ""}},
		wantMsg:  mustMakeMessage(`name:"X"`),
	}, {
		inFields: []fieldsKind{{protoreflect.MessageKind, "987654321"}},
		wantErr:  "invalid field: 987654321",
	}, {
		inFields: []fieldsKind{{protoreflect.MessageKind, "-1"}},
		wantErr:  "invalid field: -1",
	}, {
		inFields: []fieldsKind{{protoreflect.MessageKind, "k"}},
		wantErr:  "invalid field: k",
	}, {
		inFields: []fieldsKind{{protoreflect.MessageKind, "1.2"}, {protoreflect.Int32Kind, "1"}},
		wantErr:  "field 1 of int32 type cannot have sub-fields",
	}, {
		inFields: []fieldsKind{{protoreflect.Int32Kind, "1"}, {protoreflect.MessageKind, "1.2"}},
		wantErr:  "field 1 of int32 type cannot have sub-fields",
	}, {
		inFields: []fieldsKind{{protoreflect.Int32Kind, "30"}, {protoreflect.Int32Kind, "30"}},
		wantErr:  "field 30 already set as int32 type",
	}, {
		inFields: []fieldsKind{
			{protoreflect.Int32Kind, "10.20.31"},
			{protoreflect.MessageKind, "  10.20.30, 10.21   "},
			{protoreflect.GroupKind, "10"},
		},
		wantMsg: mustMakeMessage(`
			name: "X"
			field: [
				{name:"x10" number:10 label:LABEL_OPTIONAL type:TYPE_GROUP type_name:".X.X10"}
			]
			nested_type: [{
				name: "X10"
				field: [
					{name:"x20" number:20 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".X.X10.X20"},
					{name:"x21" number:21 label:LABEL_OPTIONAL type:TYPE_MESSAGE type_name:".X.X10.X21"}
				]
				nested_type: [{
					name: "X20"
					field:[
						{name:"x30" number:30 label:LABEL_OPTIONAL type:TYPE_MESSAGE, type_name:".X.X10.X20.X30"},
						{name:"x31" number:31 label:LABEL_REPEATED type:TYPE_INT32 options:{packed:true}}
					]
					nested_type: [{
						name: "X30"
					}]
				}, {
					name: "X21"
				}]
			}]
		`),
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			var fields fields
			for i, tc := range tt.inFields {
				gotErr := fields.Set(tc.fields, tc.kind)
				if gotErr != nil {
					if tt.wantErr == "" || !strings.Contains(fmt.Sprint(gotErr), tt.wantErr) {
						t.Fatalf("fields %d, Set(%q, %v) = %v, want %v", i, tc.fields, tc.kind, gotErr, tt.wantErr)
					}
					return
				}
			}
			if tt.wantErr != "" {
				t.Errorf("all Set calls succeeded, want %v error", tt.wantErr)
			}
			gotMsg := fields.messageDescriptor("X")
			if !proto.Equal(gotMsg, tt.wantMsg) {
				t.Errorf("messageDescriptor() mismatch:\ngot  %v\nwant %v", gotMsg, tt.wantMsg)
			}
			if _, err := fields.Descriptor(); err != nil {
				t.Errorf("Descriptor() = %v, want nil error", err)
			}
		})
	}
}
