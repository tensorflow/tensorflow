// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"testing"

	"google.golang.org/protobuf/proto"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
)

func TestReset(t *testing.T) {
	m := &testpb.TestAllTypes{
		OptionalSfixed64:       proto.Int64(5),
		RepeatedInt32:          []int32{},
		RepeatedFloat:          []float32{1.234, 5.678},
		MapFixed64Fixed64:      map[uint64]uint64{5: 7},
		MapStringString:        map[string]string{},
		OptionalForeignMessage: &testpb.ForeignMessage{},
		OneofField:             (*testpb.TestAllTypes_OneofUint32)(nil),
		OneofOptional:          (*testpb.TestAllTypes_OneofOptionalUint32)(nil),
	}
	m.ProtoReflect().SetUnknown([]byte{})

	proto.Reset(m)

	if m.OptionalSfixed64 != nil {
		t.Errorf("m.OptionalSfixed64 = %p, want nil", m.OptionalSfixed64)
	}
	if m.RepeatedInt32 != nil {
		t.Errorf("m.RepeatedInt32 = %p, want nil", m.RepeatedInt32)
	}
	if m.RepeatedFloat != nil {
		t.Errorf("m.RepeatedFloat = %p, want nil", m.RepeatedFloat)
	}
	if m.MapFixed64Fixed64 != nil {
		t.Errorf("m.MapFixed64Fixed64 = %p, want nil", m.MapFixed64Fixed64)
	}
	if m.MapStringString != nil {
		t.Errorf("m.MapStringString = %p, want nil", m.MapStringString)
	}
	if m.OptionalForeignMessage != nil {
		t.Errorf("m.OptionalForeignMessage = %p, want nil", m.OptionalForeignMessage)
	}
	if m.OneofField != nil {
		t.Errorf("m.OneofField = %p, want nil", m.OneofField)
	}
	if m.OneofOptional != nil {
		t.Errorf("m.OneofOptional = %p, want nil", m.OneofOptional)
	}

	if got := m.ProtoReflect().GetUnknown(); got != nil {
		t.Errorf("m.ProtoReflect().GetUnknown() = %d, want nil", got)
	}
}
