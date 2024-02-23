// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tag_test

import (
	"reflect"
	"testing"

	"google.golang.org/protobuf/internal/encoding/tag"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protodesc"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func Test(t *testing.T) {
	fd := new(filedesc.Field)
	fd.L0.ParentFile = filedesc.SurrogateProto3
	fd.L0.FullName = "foo_field"
	fd.L1.Number = 1337
	fd.L1.Cardinality = protoreflect.Repeated
	fd.L1.Kind = protoreflect.BytesKind
	fd.L1.Default = filedesc.DefaultValue(protoreflect.ValueOf([]byte("hello, \xde\xad\xbe\xef\n")), nil)

	// Marshal test.
	gotTag := tag.Marshal(fd, "")
	wantTag := `bytes,1337,rep,name=foo_field,json=fooField,proto3,def=hello, \336\255\276\357\n`
	if gotTag != wantTag {
		t.Errorf("Marshal() = `%v`, want `%v`", gotTag, wantTag)
	}

	// Unmarshal test.
	gotFD := tag.Unmarshal(wantTag, reflect.TypeOf([]byte{}), nil)
	wantFD := fd
	if !proto.Equal(protodesc.ToFieldDescriptorProto(gotFD), protodesc.ToFieldDescriptorProto(wantFD)) {
		t.Errorf("Umarshal() mismatch:\ngot  %v\nwant %v", gotFD, wantFD)
	}
}
