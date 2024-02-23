// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"testing"

	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/protobuild"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protopack"

	testpb "google.golang.org/protobuf/internal/testprotos/test"
	weakpb "google.golang.org/protobuf/internal/testprotos/test/weak1"
)

func init() {
	if flags.ProtoLegacy {
		testValidMessages = append(testValidMessages, testWeakValidMessages...)
		testInvalidMessages = append(testInvalidMessages, testWeakInvalidMessages...)
		testMerges = append(testMerges, testWeakMerges...)
	}
}

var testWeakValidMessages = []testProto{
	{
		desc: "weak message",
		decodeTo: []proto.Message{
			func() proto.Message {
				if !flags.ProtoLegacy {
					return nil
				}
				m := &testpb.TestWeak{}
				m.SetWeakMessage1(&weakpb.WeakImportMessage1{
					A: proto.Int32(1000),
				})
				m.ProtoReflect().SetUnknown(protopack.Message{
					protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
						protopack.Tag{1, protopack.VarintType}, protopack.Varint(2000),
					}),
				}.Marshal())
				return m
			}(),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(1000),
			}),
			protopack.Tag{2, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(2000),
			}),
		}.Marshal(),
	},
}

var testWeakInvalidMessages = []testProto{
	{
		desc:     "invalid field number 0 in weak message",
		decodeTo: []proto.Message{(*testpb.TestWeak)(nil)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{0, protopack.VarintType}, protopack.Varint(1000),
			}),
		}.Marshal(),
	},
}

var testWeakMerges = []testMerge{
	{
		desc: "clone weak message",
		src: protobuild.Message{
			"weak_message1": protobuild.Message{
				"a": 1,
			},
		},
		types: []proto.Message{&testpb.TestWeak{}},
	}, {
		desc: "merge weak message",
		dst: protobuild.Message{
			"weak_message1": protobuild.Message{
				"a": 1,
			},
		},
		src: protobuild.Message{
			"weak_message1": protobuild.Message{
				"a": 2,
			},
		},
		want: protobuild.Message{
			"weak_message1": protobuild.Message{
				"a": 2,
			},
		},
		types: []proto.Message{&testpb.TestWeak{}},
	},
}

func TestWeakNil(t *testing.T) {
	if !flags.ProtoLegacy {
		t.SkipNow()
	}

	m := new(testpb.TestWeak)
	if v, ok := m.GetWeakMessage1().(*weakpb.WeakImportMessage1); !ok || v != nil {
		t.Errorf("m.GetWeakMessage1() = type %[1]T(%[1]v), want (*weakpb.WeakImportMessage1)", v)
	}
}

func TestWeakMarshalNil(t *testing.T) {
	if !flags.ProtoLegacy {
		t.SkipNow()
	}

	m := new(testpb.TestWeak)
	m.SetWeakMessage1(nil)
	if b, err := proto.Marshal(m); err != nil || len(b) != 0 {
		t.Errorf("Marshal(weak field set to nil) = [%x], %v; want [], nil", b, err)
	}
	m.SetWeakMessage1((*weakpb.WeakImportMessage1)(nil))
	if b, err := proto.Marshal(m); err != nil || len(b) != 0 {
		t.Errorf("Marshal(weak field set to typed nil) = [%x], %v; want [], nil", b, err)
	}
}
