// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package wirefuzz includes a fuzzer for the wire marshaler and unmarshaler.
package wirefuzz

import (
	"fmt"

	"google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoregistry"
	piface "google.golang.org/protobuf/runtime/protoiface"

	fuzzpb "google.golang.org/protobuf/internal/testprotos/fuzz"
)

// Fuzz is a fuzzer for proto.Marshal and proto.Unmarshal.
func Fuzz(data []byte) (score int) {
	// Unmarshal and Validate should agree about the validity of the message.
	m1 := &fuzzpb.Fuzz{}
	mt := m1.ProtoReflect().Type()
	_, valid := impl.Validate(mt, piface.UnmarshalInput{Buf: data})
	if err := (proto.UnmarshalOptions{AllowPartial: true}).Unmarshal(data, m1); err != nil {
		switch valid {
		case impl.ValidationUnknown:
		case impl.ValidationInvalid:
		default:
			panic("unmarshal error with validation status: " + valid.String())
		}
		return 0
	}
	switch valid {
	case impl.ValidationUnknown:
	case impl.ValidationValid:
	default:
		panic("unmarshal ok with validation status: " + valid.String())
	}

	// Unmarshal, Validate, and CheckInitialized should agree about initialization.
	checkInit := proto.CheckInitialized(m1) == nil
	methods := m1.ProtoReflect().ProtoMethods()
	in := piface.UnmarshalInput{Message: mt.New(), Resolver: protoregistry.GlobalTypes, Depth: 10000}
	if checkInit {
		// If the message initialized, the both Unmarshal and Validate should
		// report it as such. False negatives are tolerated, but have a
		// significant impact on performance. In general, they should always
		// properly determine initialization for any normalized message,
		// we produce by re-marshaling the message.
		in.Buf, _ = proto.Marshal(m1)
		if out, _ := methods.Unmarshal(in); out.Flags&piface.UnmarshalInitialized == 0 {
			panic("unmarshal reports initialized message as partial")
		}
		if out, _ := impl.Validate(mt, in); out.Flags&piface.UnmarshalInitialized == 0 {
			panic("validate reports initialized message as partial")
		}
	} else {
		// If the message is partial, then neither Unmarshal nor Validate
		// should ever report it as such. False positives are unacceptable.
		in.Buf = data
		if out, _ := methods.Unmarshal(in); out.Flags&piface.UnmarshalInitialized != 0 {
			panic("unmarshal reports partial message as initialized")
		}
		if out, _ := impl.Validate(mt, in); out.Flags&piface.UnmarshalInitialized != 0 {
			panic("validate reports partial message as initialized")
		}
	}

	// Round-trip Marshal and Unmarshal should produce the same messages.
	data1, err := proto.MarshalOptions{AllowPartial: !checkInit}.Marshal(m1)
	if err != nil {
		panic(err)
	}
	if proto.Size(m1) != len(data1) {
		panic(fmt.Errorf("size does not match output: %d != %d", proto.Size(m1), len(data1)))
	}
	m2 := &fuzzpb.Fuzz{}
	if err := (proto.UnmarshalOptions{AllowPartial: !checkInit}).Unmarshal(data1, m2); err != nil {
		panic(err)
	}
	if !proto.Equal(m1, m2) {
		panic("not equal")
	}
	return 1
}
