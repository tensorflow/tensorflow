// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"testing"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	retentionpb "google.golang.org/protobuf/cmd/protoc-gen-go/testdata/retention"
)

func TestFileOptionRetention(t *testing.T) {
	options := retentionpb.File_cmd_protoc_gen_go_testdata_retention_retention_proto.Options()
	tests := []struct {
		name      string
		ext       protoreflect.ExtensionType
		wantField bool
		wantValue int32
	}{
		{
			name:      "imported_plain_option",
			ext:       retentionpb.E_ImportedPlainOption,
			wantField: true,
			wantValue: 1,
		},
		{
			name:      "imported_runtime_option",
			ext:       retentionpb.E_ImportedRuntimeRetentionOption,
			wantField: true,
			wantValue: 2,
		},
		{
			name:      "imported_source_option",
			ext:       retentionpb.E_ImportedSourceRetentionOption,
			wantField: false,
			wantValue: 0,
		},
		{
			name:      "plain_option",
			ext:       retentionpb.E_PlainOption,
			wantField: true,
			wantValue: 1,
		},
		{
			name:      "runtime_option",
			ext:       retentionpb.E_RuntimeRetentionOption,
			wantField: true,
			wantValue: 2,
		},
		{
			name:      "source_option",
			ext:       retentionpb.E_SourceRetentionOption,
			wantField: false,
			wantValue: 0,
		},
	}

	for _, test := range tests {
		if test.wantField != proto.HasExtension(options, test.ext) {
			t.Errorf("HasExtension(%s): got %v, want %v", test.name, proto.HasExtension(options, test.ext), test.wantField)
		}
		if test.wantValue != proto.GetExtension(options, test.ext).(int32) {
			t.Errorf("GetExtension(%s): got %d, want %d", test.name, proto.GetExtension(options, test.ext).(int32), test.wantValue)
		}
	}
}

func TestAllEntitiesWithMessageOption(t *testing.T) {
	file := retentionpb.File_cmd_protoc_gen_go_testdata_retention_retention_proto
	verifyDescriptorOptions(t, string(file.Name()), file.Options())
	verifyEnums(t, file.Enums())
	verifyMessages(t, file.Messages())
	verifyExtensions(t, file.Extensions())
	verifyServices(t, file.Services())
}

func verifyExtensions(t *testing.T, extensions protoreflect.ExtensionDescriptors) {
	t.Helper()
	for i := 0; i < extensions.Len(); i++ {
		verifyDescriptorOptions(t, string(extensions.Get(i).Name()), extensions.Get(i).Options())
	}
}

func verifyMessages(t *testing.T, messages protoreflect.MessageDescriptors) {
	t.Helper()
	for i := 0; i < messages.Len(); i++ {
		verifyDescriptorOptions(t, string(messages.Get(i).Name()), messages.Get(i).Options())
		verifyEnums(t, messages.Get(i).Enums())
		verifyMessages(t, messages.Get(i).Messages())
		verifyExtensions(t, messages.Get(i).Extensions())
		verifyFields(t, messages.Get(i).Fields())
	}
}

func verifyFields(t *testing.T, fields protoreflect.FieldDescriptors) {
	t.Helper()
	for i := 0; i < fields.Len(); i++ {
		verifyDescriptorOptions(t, string(fields.Get(i).Name()), fields.Get(i).Options())
	}
}

func verifyEnums(t *testing.T, enums protoreflect.EnumDescriptors) {
	t.Helper()
	for i := 0; i < enums.Len(); i++ {
		verifyDescriptorOptions(t, string(enums.Get(i).Name()), enums.Get(i).Options())
		verifyEnumValues(t, enums.Get(i).Values())
	}
}

func verifyEnumValues(t *testing.T, values protoreflect.EnumValueDescriptors) {
	t.Helper()
	for i := 0; i < values.Len(); i++ {
		verifyDescriptorOptions(t, string(values.Get(i).Name()), values.Get(i).Options())
	}
}

func verifyServices(t *testing.T, services protoreflect.ServiceDescriptors) {
	t.Helper()
	for i := 0; i < services.Len(); i++ {
		verifyDescriptorOptions(t, string(services.Get(i).Name()), services.Get(i).Options())
		verifyMethods(t, services.Get(i).Methods())
	}
}

func verifyMethods(t *testing.T, methods protoreflect.MethodDescriptors) {
	t.Helper()
	for i := 0; i < methods.Len(); i++ {
		verifyDescriptorOptions(t, string(methods.Get(i).Name()), methods.Get(i).Options())
	}
}

func verifyDescriptorOptions(t *testing.T, entity string, options protoreflect.ProtoMessage) {
	t.Helper()
	options.ProtoReflect().Range(func(fd protoreflect.FieldDescriptor, v protoreflect.Value) bool {
		maybeVerifyOption(t, fd, v)
		return true
	})
}

func maybeVerifyOption(t *testing.T, fd protoreflect.FieldDescriptor, v protoreflect.Value) {
	t.Helper()
	if fd.Kind() == protoreflect.MessageKind && string(fd.Message().FullName()) == "goproto.proto.testretention.OptionsMessage" {
		if fd.IsList() {
			for i := 0; i < v.List().Len(); i++ {
				verifyOptionsMessage(t, string(fd.FullName()), v.List().Get(i).Message().Interface().(*retentionpb.OptionsMessage))
			}
		} else {
			verifyOptionsMessage(t, string(fd.FullName()), v.Message().Interface().(*retentionpb.OptionsMessage))
		}
	}
}

func verifyOptionsMessage(t *testing.T, entity string, msg *retentionpb.OptionsMessage) {
	t.Helper()
	if msg.PlainField == nil {
		t.Errorf("%s.OptionsMessage.HasField(plain_field): got false, want true", entity)
	}
	if msg.GetPlainField() != 1 {
		t.Errorf("%s.OptionsMessage.GetField(plain_field): got %d, want 1", entity, msg.GetPlainField())
	}
	if msg.RuntimeRetentionField == nil {
		t.Errorf("%s.OptionsMessage.HasField(runtime_retention_field): got false, want true", entity)
	}
	if msg.GetRuntimeRetentionField() != 2 {
		t.Errorf("%s.OptionsMessage.GetField(runtime_retention_field): got %d, want 2", entity, msg.GetRuntimeRetentionField())
	}
	if msg.SourceRetentionField != nil {
		t.Errorf("%s.OptionsMessage.HasField(source_retention_field): got true, want false", entity)
	}
	if msg.GetSourceRetentionField() != 0 {
		// Checking that we get 0 even though this was set to 3 in the source file
		t.Errorf("%s.OptionsMessage.GetField(source_retention_field): got %d, want 0", entity, msg.GetSourceRetentionField())
	}
}
