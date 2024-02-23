// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dynamicpb_test

import (
	"strings"
	"testing"

	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/types/descriptorpb"
	"google.golang.org/protobuf/types/dynamicpb"

	registrypb "google.golang.org/protobuf/internal/testprotos/registry"
)

var _ protoregistry.ExtensionTypeResolver = &dynamicpb.Types{}
var _ protoregistry.MessageTypeResolver = &dynamicpb.Types{}

func newTestTypes() *dynamicpb.Types {
	files := &protoregistry.Files{}
	files.RegisterFile(registrypb.File_internal_testprotos_registry_test_proto)
	return dynamicpb.NewTypes(files)
}

func TestDynamicTypesTypeMismatch(t *testing.T) {
	types := newTestTypes()
	const messageName = "testprotos.Message1"
	const enumName = "testprotos.Enum1"

	_, err := types.FindEnumByName(messageName)
	want := "found wrong type: got message, want enum"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("types.FindEnumByName(%q) = _, %q, want %q", messageName, err, want)
	}

	_, err = types.FindMessageByName(enumName)
	want = "found wrong type: got enum, want message"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("types.FindMessageByName(%q) = _, %q, want %q", messageName, err, want)
	}

	_, err = types.FindExtensionByName(enumName)
	want = "found wrong type: got enum, want extension"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("types.FindExtensionByName(%q) = _, %q, want %q", messageName, err, want)
	}
}

func TestDynamicTypesEnumNotFound(t *testing.T) {
	types := newTestTypes()
	for _, name := range []protoreflect.FullName{
		"Enum1",
		"testprotos.DoesNotExist",
	} {
		_, err := types.FindEnumByName(name)
		if err != protoregistry.NotFound {
			t.Errorf("types.FindEnumByName(%q) = _, %v; want protoregistry.NotFound", name, err)
		}
	}
}

func TestDynamicTypesFindEnumByName(t *testing.T) {
	types := newTestTypes()
	name := protoreflect.FullName("testprotos.Enum1")
	et, err := types.FindEnumByName(name)
	if err != nil {
		t.Fatalf("types.FindEnumByName(%q) = %v", name, err)
	}
	if got, want := et.Descriptor().FullName(), name; got != want {
		t.Fatalf("types.FindEnumByName(%q).Descriptor().FullName() = %q, want %q", name, got, want)
	}
}

func TestDynamicTypesMessageNotFound(t *testing.T) {
	types := newTestTypes()
	for _, name := range []protoreflect.FullName{
		"Message1",
		"testprotos.DoesNotExist",
	} {
		_, err := types.FindMessageByName(name)
		if err != protoregistry.NotFound {
			t.Errorf("types.FindMessageByName(%q) = _, %v; want protoregistry.NotFound", name, err)
		}
	}
}

func TestDynamicTypesFindMessageByName(t *testing.T) {
	types := newTestTypes()
	name := protoreflect.FullName("testprotos.Message1")
	mt, err := types.FindMessageByName(name)
	if err != nil {
		t.Fatalf("types.FindMessageByName(%q) = %v", name, err)
	}
	if got, want := mt.Descriptor().FullName(), name; got != want {
		t.Fatalf("types.FindMessageByName(%q).Descriptor().FullName() = %q, want %q", name, got, want)
	}
}

func TestDynamicTypesExtensionNotFound(t *testing.T) {
	types := newTestTypes()
	for _, name := range []protoreflect.FullName{
		"string_field",
		"testprotos.DoesNotExist",
	} {
		_, err := types.FindExtensionByName(name)
		if err != protoregistry.NotFound {
			t.Errorf("types.FindExtensionByName(%q) = _, %v; want protoregistry.NotFound", name, err)
		}
	}
	messageName := protoreflect.FullName("testprotos.Message1")
	if _, err := types.FindExtensionByNumber(messageName, 100); err != protoregistry.NotFound {
		t.Errorf("types.FindExtensionByNumber(%q, 100) = _, %v; want protoregistry.NotFound", messageName, 100)
	}
}

func TestDynamicTypesFindExtensionByNameOrNumber(t *testing.T) {
	types := newTestTypes()
	messageName := protoreflect.FullName("testprotos.Message1")
	mt, err := types.FindMessageByName(messageName)
	if err != nil {
		t.Fatalf("types.FindMessageByName(%q) = %v", messageName, err)
	}
	for _, extensionName := range []protoreflect.FullName{
		"testprotos.string_field",
		"testprotos.Message4.message_field",
	} {
		xt, err := types.FindExtensionByName(extensionName)
		if err != nil {
			t.Fatalf("types.FindExtensionByName(%q) = %v", extensionName, err)
		}
		if got, want := xt.TypeDescriptor().FullName(), extensionName; got != want {
			t.Fatalf("types.FindExtensionByName(%q).TypeDescriptor().FullName() = %q, want %q", extensionName, got, want)
		}
		if got, want := xt.TypeDescriptor().ContainingMessage(), mt.Descriptor(); got != want {
			t.Fatalf("xt.TypeDescriptor().ContainingMessage() = %q, want %q", got.FullName(), want.FullName())
		}
		number := xt.TypeDescriptor().Number()
		xt2, err := types.FindExtensionByNumber(messageName, number)
		if err != nil {
			t.Fatalf("types.FindExtensionByNumber(%q, %v) = %v", messageName, number, err)
		}
		if xt != xt2 {
			t.Fatalf("FindExtensionByName returned a differet extension than FindExtensionByNumber")
		}
	}
}

func TestDynamicTypesFilesChangeAfterCreation(t *testing.T) {
	files := &protoregistry.Files{}
	files.RegisterFile(descriptorpb.File_google_protobuf_descriptor_proto)
	types := dynamicpb.NewTypes(files)

	// Not found: Files registry does not contain this file.
	const message = "testprotos.Message1"
	const number = 11
	if _, err := types.FindMessageByName(message); err != protoregistry.NotFound {
		t.Errorf("types.FindMessageByName(%q) = %v, want protoregistry.NotFound", message, err)
	}
	if _, err := types.FindExtensionByNumber(message, number); err != protoregistry.NotFound {
		t.Errorf("types.FindExtensionByNumber(%q, %v) = %v, want protoregistry.NotFound", message, number, err)
	}

	// Found: Add the file to the registry and recheck.
	files.RegisterFile(registrypb.File_internal_testprotos_registry_test_proto)
	if _, err := types.FindMessageByName(message); err != nil {
		t.Errorf("types.FindMessageByName(%q) = %v, want nil", message, err)
	}
	if _, err := types.FindExtensionByNumber(message, number); err != nil {
		t.Errorf("types.FindExtensionByNumber(%q, %v) = %v, want nil", message, number, err)
	}
}
