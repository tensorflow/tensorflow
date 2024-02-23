// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go native fuzzing was added in go1.18. Remove this once we stop supporting
// go1.17.
//go:build go1.18

package proto_test

import (
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	testfuzzpb "google.golang.org/protobuf/internal/testprotos/editionsfuzztest"
)

func TestUnmarshalInvalidGroupField(t *testing.T) {
	in := []byte("\x82\x01\x010")
	// Test proto2 proto
	proto2Proto := &testfuzzpb.TestAllTypesProto2{}

	if err := proto.Unmarshal(in, proto2Proto); err != nil {
		t.Error(err)
	}
	// Test equivalent editions proto
	editionsProto := &testfuzzpb.TestAllTypesProto2Editions{}

	if err := proto.Unmarshal(in, editionsProto); err != nil {
		t.Error(err)
	}
}

// compareEquivalentProtos compares equivalent messages m0 and m1, where one is
// typically a Protobuf Editions message and the other isn't. It unmarshals the
// wireBytes into a message of type m0 and one of type m1 and compares the
// resulting messages for equality (ignoring type names). m0 and m1 must
// describe equivalent messages, meaning having the same field numbers and
// types.
func compareEquivalentProtos(t *testing.T, wireBytes []byte, m0, m1 proto.Message) {
	t.Helper()
	m0Instance := m0.ProtoReflect().Type().New().Interface()
	errM0 := proto.Unmarshal(wireBytes, m0Instance)
	m1Instance := m1.ProtoReflect().Type().New().Interface()
	errM1 := proto.Unmarshal(wireBytes, m1Instance)

	// Check that the error are the same (possible nil)
	errorsMatch := (errM1 != nil) == (errM0 != nil)
	if errM1 != nil && errM0 != nil {
		errorsMatch = errM1.Error() == errM0.Error()
	}
	if !errorsMatch {
		t.Fatalf("errors not equal:\n%T error: %v\n%T error:%v", m0, errM0, m1, errM1)
	}

	// Marshal the editions proto and unmarshal it into the equivalent proto2
	// message to be able to compare the messages.
	// This tests slightly more than necessary but should only lead to more
	// coverage (unless the marshalling would undo errors of the unmarshalling
	// which is very unlikely).
	roundTrippedM0 := m0.ProtoReflect().Type().New().Interface()
	err := roundTripMessage(roundTrippedM0, m1Instance)
	if err != nil {
		t.Fatalf("failed round tripping proto: %v", err)
	}

	// The cmp package does not deal with NaN on its own and will report
	// NaN != NaN.
	optNaN64 := cmp.Comparer(func(x, y float32) bool {
		return (math.IsNaN(float64(x)) && math.IsNaN(float64(y))) || x == y
	})
	optNaN32 := cmp.Comparer(func(x, y float64) bool {
		return (math.IsNaN(x) && math.IsNaN(y)) || x == y
	})
	if diff := cmp.Diff(m0Instance, roundTrippedM0, protocmp.Transform(), optNaN64, optNaN32); diff != "" {
		t.Error(diff)
	}

	if sizeM0, sizeM1 := proto.Size(m0Instance), proto.Size(m1Instance); sizeM0 != sizeM1 {
		t.Errorf("proto.Size() not equal:\n%T size = %v\n%T size = %v", m0, sizeM0, m1, sizeM1)
	}
}

func FuzzProto2EditionConversion(f *testing.F) {
	f.Add([]byte("Hello World!"))
	f.Fuzz(func(t *testing.T, in []byte) {
		compareEquivalentProtos(t, in, (*testfuzzpb.TestAllTypesProto2)(nil), (*testfuzzpb.TestAllTypesProto2Editions)(nil))
	})
}

func FuzzProto3EditionConversion(f *testing.F) {
	f.Add([]byte("Hello World!"))
	f.Fuzz(func(t *testing.T, in []byte) {
		compareEquivalentProtos(t, in, (*testfuzzpb.TestAllTypesProto3)(nil), (*testfuzzpb.TestAllTypesProto3Editions)(nil))
	})
}
