// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl_test

import (
	"bytes"
	"compress/gzip"
	"io/ioutil"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/internal/pragma"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"

	proto2_20160225 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20160225_2fc053c5"
	proto2_20160519 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20160519_a4ab9ec5"
	proto2_20180125 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20180125_92554152"
	proto2_20180430 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20180430_b4deda09"
	proto2_20180814 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20180814_aa810b61"
	proto2_20190205 "google.golang.org/protobuf/internal/testprotos/legacy/proto2_20190205_c823c79e"
	proto3_20160225 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20160225_2fc053c5"
	proto3_20160519 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20160519_a4ab9ec5"
	proto3_20180125 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20180125_92554152"
	proto3_20180430 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20180430_b4deda09"
	proto3_20180814 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20180814_aa810b61"
	proto3_20190205 "google.golang.org/protobuf/internal/testprotos/legacy/proto3_20190205_c823c79e"
	"google.golang.org/protobuf/types/descriptorpb"
)

func mustLoadFileDesc(b []byte, _ []int) protoreflect.FileDescriptor {
	zr, err := gzip.NewReader(bytes.NewReader(b))
	if err != nil {
		panic(err)
	}
	b, err = ioutil.ReadAll(zr)
	if err != nil {
		panic(err)
	}
	p := new(descriptorpb.FileDescriptorProto)
	err = proto.UnmarshalOptions{DiscardUnknown: true}.Unmarshal(b, p)
	if err != nil {
		panic(err)
	}
	fd, err := protodesc.NewFile(p, nil)
	if err != nil {
		panic(err)
	}
	return fd
}

func TestDescriptor(t *testing.T) {
	var tests []struct{ got, want protoreflect.Descriptor }

	fileDescP2_20160225 := mustLoadFileDesc(new(proto2_20160225.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20160225.SiblingEnum(0))),
		want: fileDescP2_20160225.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20160225.Message_ChildEnum(0))),
		want: fileDescP2_20160225.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.SiblingMessage))),
		want: fileDescP2_20160225.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_ChildMessage))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message))),
		want: fileDescP2_20160225.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_NamedGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_OptionalGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_RequiredGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_RepeatedGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_OneofGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160225.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20160225.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20160225 := mustLoadFileDesc(new(proto3_20160225.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20160225.SiblingEnum(0))),
		want: fileDescP3_20160225.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20160225.Message_ChildEnum(0))),
		want: fileDescP3_20160225.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160225.SiblingMessage))),
		want: fileDescP3_20160225.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160225.Message_ChildMessage))),
		want: fileDescP3_20160225.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160225.Message))),
		want: fileDescP3_20160225.Messages().ByName("Message"),
	}}...)

	fileDescP2_20160519 := mustLoadFileDesc(new(proto2_20160519.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20160519.SiblingEnum(0))),
		want: fileDescP2_20160519.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20160519.Message_ChildEnum(0))),
		want: fileDescP2_20160519.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.SiblingMessage))),
		want: fileDescP2_20160519.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_ChildMessage))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message))),
		want: fileDescP2_20160519.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_NamedGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_OptionalGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_RequiredGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_RepeatedGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_OneofGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20160519.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20160519.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20160519 := mustLoadFileDesc(new(proto3_20160519.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20160519.SiblingEnum(0))),
		want: fileDescP3_20160519.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20160519.Message_ChildEnum(0))),
		want: fileDescP3_20160519.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160519.SiblingMessage))),
		want: fileDescP3_20160519.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160519.Message_ChildMessage))),
		want: fileDescP3_20160519.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20160519.Message))),
		want: fileDescP3_20160519.Messages().ByName("Message"),
	}}...)

	fileDescP2_20180125 := mustLoadFileDesc(new(proto2_20180125.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180125.SiblingEnum(0))),
		want: fileDescP2_20180125.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180125.Message_ChildEnum(0))),
		want: fileDescP2_20180125.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.SiblingMessage))),
		want: fileDescP2_20180125.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_ChildMessage))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message))),
		want: fileDescP2_20180125.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_NamedGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_OptionalGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_RequiredGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_RepeatedGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_OneofGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180125.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20180125.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20180125 := mustLoadFileDesc(new(proto3_20180125.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180125.SiblingEnum(0))),
		want: fileDescP3_20180125.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180125.Message_ChildEnum(0))),
		want: fileDescP3_20180125.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180125.SiblingMessage))),
		want: fileDescP3_20180125.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180125.Message_ChildMessage))),
		want: fileDescP3_20180125.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180125.Message))),
		want: fileDescP3_20180125.Messages().ByName("Message"),
	}}...)

	fileDescP2_20180430 := mustLoadFileDesc(new(proto2_20180430.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180430.SiblingEnum(0))),
		want: fileDescP2_20180430.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180430.Message_ChildEnum(0))),
		want: fileDescP2_20180430.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.SiblingMessage))),
		want: fileDescP2_20180430.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_ChildMessage))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message))),
		want: fileDescP2_20180430.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_NamedGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_OptionalGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_RequiredGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_RepeatedGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_OneofGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180430.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20180430.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20180430 := mustLoadFileDesc(new(proto3_20180430.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180430.SiblingEnum(0))),
		want: fileDescP3_20180430.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180430.Message_ChildEnum(0))),
		want: fileDescP3_20180430.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180430.SiblingMessage))),
		want: fileDescP3_20180430.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180430.Message_ChildMessage))),
		want: fileDescP3_20180430.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180430.Message))),
		want: fileDescP3_20180430.Messages().ByName("Message"),
	}}...)

	fileDescP2_20180814 := mustLoadFileDesc(new(proto2_20180814.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180814.SiblingEnum(0))),
		want: fileDescP2_20180814.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20180814.Message_ChildEnum(0))),
		want: fileDescP2_20180814.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.SiblingMessage))),
		want: fileDescP2_20180814.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_ChildMessage))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message))),
		want: fileDescP2_20180814.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_NamedGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_OptionalGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_RequiredGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_RepeatedGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_OneofGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20180814.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20180814.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20180814 := mustLoadFileDesc(new(proto3_20180814.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180814.SiblingEnum(0))),
		want: fileDescP3_20180814.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20180814.Message_ChildEnum(0))),
		want: fileDescP3_20180814.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180814.SiblingMessage))),
		want: fileDescP3_20180814.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180814.Message_ChildMessage))),
		want: fileDescP3_20180814.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20180814.Message))),
		want: fileDescP3_20180814.Messages().ByName("Message"),
	}}...)

	fileDescP2_20190205 := mustLoadFileDesc(new(proto2_20190205.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20190205.SiblingEnum(0))),
		want: fileDescP2_20190205.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto2_20190205.Message_ChildEnum(0))),
		want: fileDescP2_20190205.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.SiblingMessage))),
		want: fileDescP2_20190205.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_ChildMessage))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message))),
		want: fileDescP2_20190205.Messages().ByName("Message"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_NamedGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("NamedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_OptionalGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("OptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_RequiredGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("RequiredGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_RepeatedGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("RepeatedGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_OneofGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("OneofGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_ExtensionOptionalGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("ExtensionOptionalGroup"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto2_20190205.Message_ExtensionRepeatedGroup))),
		want: fileDescP2_20190205.Messages().ByName("Message").Messages().ByName("ExtensionRepeatedGroup"),
	}}...)

	fileDescP3_20190205 := mustLoadFileDesc(new(proto3_20190205.Message).Descriptor())
	tests = append(tests, []struct{ got, want protoreflect.Descriptor }{{
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20190205.SiblingEnum(0))),
		want: fileDescP3_20190205.Enums().ByName("SiblingEnum"),
	}, {
		got:  impl.LegacyLoadEnumDesc(reflect.TypeOf(proto3_20190205.Message_ChildEnum(0))),
		want: fileDescP3_20190205.Messages().ByName("Message").Enums().ByName("ChildEnum"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20190205.SiblingMessage))),
		want: fileDescP3_20190205.Messages().ByName("SiblingMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20190205.Message_ChildMessage))),
		want: fileDescP3_20190205.Messages().ByName("Message").Messages().ByName("ChildMessage"),
	}, {
		got:  impl.LegacyLoadMessageDesc(reflect.TypeOf(new(proto3_20190205.Message))),
		want: fileDescP3_20190205.Messages().ByName("Message"),
	}}...)

	// TODO: We need a test package to compare descriptors.
	type list interface {
		Len() int
		pragma.DoNotImplement
	}
	opts := cmp.Options{
		cmp.Transformer("", func(x list) []interface{} {
			out := make([]interface{}, x.Len())
			v := reflect.ValueOf(x)
			for i := 0; i < x.Len(); i++ {
				m := v.MethodByName("Get")
				out[i] = m.Call([]reflect.Value{reflect.ValueOf(i)})[0].Interface()
			}
			return out
		}),
		cmp.Transformer("", func(x protoreflect.Descriptor) map[string]interface{} {
			out := make(map[string]interface{})
			v := reflect.ValueOf(x)
			for i := 0; i < v.NumMethod(); i++ {
				name := v.Type().Method(i).Name
				if m := v.Method(i); m.Type().NumIn() == 0 && m.Type().NumOut() == 1 {
					switch name {
					case "ParentFile", "Parent":
						// Ignore parents to avoid recursive cycle.
					case "Index":
						// Ignore index since legacy descriptors have no parent.
					case "Options":
						// Ignore descriptor options since protos are not cmperable.
					case "Enums", "Messages", "Extensions":
						// Ignore nested message and enum declarations since
						// legacy descriptors are all created standalone.
					case "HasJSONName":
						// Ignore this since the semantics of the field has
						// changed across protoc and protoc-gen-go releases.
					case "ContainingOneof", "ContainingMessage", "Enum", "Message":
						// Avoid descending into a dependency to avoid a cycle.
						// Just record the full name if available.
						//
						// TODO: Cycle support in cmp would be useful here.
						v := m.Call(nil)[0]
						if !v.IsNil() {
							out[name] = v.Interface().(protoreflect.Descriptor).FullName()
						}
					default:
						out[name] = m.Call(nil)[0].Interface()
					}
				}
			}
			return out
		}),
		cmp.Transformer("", func(v protoreflect.Value) interface{} {
			return v.Interface()
		}),
	}

	for _, tt := range tests {
		t.Run(string(tt.want.FullName()), func(t *testing.T) {
			if diff := cmp.Diff(&tt.want, &tt.got, opts); diff != "" {
				t.Errorf("descriptor mismatch (-want, +got):\n%s", diff)
			}
		})
	}
}
