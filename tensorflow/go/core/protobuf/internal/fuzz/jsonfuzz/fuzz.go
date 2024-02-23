// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package jsonfuzz includes fuzzers for protojson.Marshal and protojson.Unmarshal.
package jsonfuzz

import (
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"

	fuzzpb "google.golang.org/protobuf/internal/testprotos/fuzz"
)

// Fuzz is a fuzzer for proto.Marshal and proto.Unmarshal.
func Fuzz(data []byte) (score int) {
	m1 := &fuzzpb.Fuzz{}
	if err := (protojson.UnmarshalOptions{
		AllowPartial: true,
	}).Unmarshal(data, m1); err != nil {
		return 0
	}
	data1, err := protojson.MarshalOptions{
		AllowPartial: true,
	}.Marshal(m1)
	if err != nil {
		panic(err)
	}
	m2 := &fuzzpb.Fuzz{}
	if err := (protojson.UnmarshalOptions{
		AllowPartial: true,
	}).Unmarshal(data1, m2); err != nil {
		return 0
	}
	if !proto.Equal(m1, m2) {
		panic("not equal")
	}
	return 1
}
