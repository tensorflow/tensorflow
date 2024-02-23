// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package prototest

import (
	"testing"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// Enum tests an [protoreflect.EnumType] implementation.
type Enum struct{}

func (test Enum) Test(t testing.TB, et protoreflect.EnumType) {
	ed := et.Descriptor()
	values := ed.Values()
	for i := 0; i < values.Len(); i++ {
		evd := values.Get(i)
		num := evd.Number()
		e := et.New(num)
		if e.Descriptor() != ed {
			t.Errorf("enumType.New(%v).Descriptor() != enumType.Descriptor(), should match", num)
		}
		if e.Type() != et {
			t.Errorf("enumType.New(%v).Type() != enumType, should match", num)
		}
		if got, want := e.Number(), num; got != want {
			t.Errorf("enumType.New(%v).Number() = %v, want %v", num, got, want)
		}
	}
}
