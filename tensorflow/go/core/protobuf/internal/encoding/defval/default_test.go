// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package defval_test

import (
	"math"
	"reflect"
	"testing"

	"google.golang.org/protobuf/internal/encoding/defval"
	"google.golang.org/protobuf/internal/filedesc"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func Test(t *testing.T) {
	evs := filedesc.EnumValues{List: []filedesc.EnumValue{{}}}
	evs.List[0].L0.ParentFile = filedesc.SurrogateProto2
	evs.List[0].L0.FullName = "ALPHA"
	evs.List[0].L1.Number = 1

	V := protoreflect.ValueOf
	tests := []struct {
		val   protoreflect.Value
		enum  protoreflect.EnumValueDescriptor
		enums protoreflect.EnumValueDescriptors
		kind  protoreflect.Kind
		strPB string
		strGo string
	}{{
		val:   V(bool(true)),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.BoolKind,
		strPB: "true",
		strGo: "1",
	}, {
		val:   V(int32(-0x1234)),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.Int32Kind,
		strPB: "-4660",
		strGo: "-4660",
	}, {
		val:   V(float32(math.Pi)),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.FloatKind,
		strPB: "3.1415927",
		strGo: "3.1415927",
	}, {
		val:   V(float64(math.Pi)),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.DoubleKind,
		strPB: "3.141592653589793",
		strGo: "3.141592653589793",
	}, {
		val:   V(string("hello, \xde\xad\xbe\xef\n")),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.StringKind,
		strPB: "hello, \xde\xad\xbe\xef\n",
		strGo: "hello, \xde\xad\xbe\xef\n",
	}, {
		val:   V([]byte("hello, \xde\xad\xbe\xef\n")),
		enum:  nil,
		enums: nil,
		kind:  protoreflect.BytesKind,
		strPB: "hello, \\336\\255\\276\\357\\n",
		strGo: "hello, \\336\\255\\276\\357\\n",
	}, {
		val:   V(protoreflect.EnumNumber(1)),
		enum:  &evs.List[0],
		enums: &evs,
		kind:  protoreflect.EnumKind,
		strPB: "ALPHA",
		strGo: "1",
	}}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			gotStr, _ := defval.Marshal(tt.val, tt.enum, tt.kind, defval.Descriptor)
			if gotStr != tt.strPB {
				t.Errorf("Marshal(%v, %v, Descriptor) = %q, want %q", tt.val, tt.kind, gotStr, tt.strPB)
			}

			gotStr, _ = defval.Marshal(tt.val, tt.enum, tt.kind, defval.GoTag)
			if gotStr != tt.strGo {
				t.Errorf("Marshal(%v, %v, GoTag) = %q, want %q", tt.val, tt.kind, gotStr, tt.strGo)
			}

			gotVal, gotEnum, _ := defval.Unmarshal(tt.strPB, tt.kind, tt.enums, defval.Descriptor)
			if !reflect.DeepEqual(gotVal.Interface(), tt.val.Interface()) || gotEnum != tt.enum {
				t.Errorf("Unmarshal(%v, %v, Descriptor) = (%q, %v), want (%q, %v)", tt.strPB, tt.kind, gotVal, gotEnum, tt.val, tt.enum)
			}

			gotVal, gotEnum, _ = defval.Unmarshal(tt.strGo, tt.kind, tt.enums, defval.GoTag)
			if !reflect.DeepEqual(gotVal.Interface(), tt.val.Interface()) || gotEnum != tt.enum {
				t.Errorf("Unmarshal(%v, %v, GoTag) = (%q, %v), want (%q, %v)", tt.strGo, tt.kind, gotVal, gotEnum, tt.val, tt.enum)
			}
		})
	}
}
