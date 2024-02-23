// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto_test

import (
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protopack"

	messagesetpb "google.golang.org/protobuf/internal/testprotos/messageset/messagesetpb"
	msetextpb "google.golang.org/protobuf/internal/testprotos/messageset/msetextpb"
)

func init() {
	if flags.ProtoLegacy {
		testValidMessages = append(testValidMessages, messageSetTestProtos...)
		testInvalidMessages = append(testInvalidMessages, messageSetInvalidTestProtos...)
	}
}

var messageSetTestProtos = []testProto{
	{
		desc: "MessageSet type_id before message content",
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
			})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
				}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc: "MessageSet type_id after message content",
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
			})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
				}),
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc: "MessageSet does not preserve unknown field",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
			extend(msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
			}),
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
			}),
			protopack.Tag{1, protopack.EndGroupType},
			// Unknown field
			protopack.Tag{4, protopack.VarintType}, protopack.Varint(30),
		}.Marshal(),
	},
	{
		desc: "MessageSet with unknown type_id",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
			unknown(protopack.Message{
				protopack.Tag{999, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
				}),
			}.Marshal()),
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(999),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
			}),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "MessageSet merges repeated message fields in item",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
			extend(msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
				Ext1Field2: proto.Int32(20),
			}),
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
			}),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(20),
			}),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "MessageSet merges message fields in repeated items",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
			extend(msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
				Ext1Field2: proto.Int32(20),
			}),
			extend(msetextpb.E_Ext2_MessageSetExtension, &msetextpb.Ext2{
				Ext2Field1: proto.Int32(30),
			}),
		)},
		wire: protopack.Message{
			// Ext1, field1
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
			}),
			protopack.Tag{1, protopack.EndGroupType},
			// Ext2, field1
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1001),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(30),
			}),
			protopack.Tag{1, protopack.EndGroupType},
			// Ext2, field2
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(20),
			}),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "MessageSet with missing type_id",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
			}),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "MessageSet with missing message",
		decodeTo: []proto.Message{build(
			&messagesetpb.MessageSet{},
			extend(msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{}),
		)},
		wire: protopack.Message{
			protopack.Tag{1, protopack.StartGroupType},
			protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
			protopack.Tag{1, protopack.EndGroupType},
		}.Marshal(),
	},
	{
		desc: "MessageSet with type id out of valid field number range",
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_ExtLargeNumber_MessageSetExtension, &msetextpb.ExtLargeNumber{})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(protowire.MaxValidNumber + 1),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc: "MessageSet with unknown type id out of valid field number range",
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			m.MessageSet.ProtoReflect().SetUnknown(
				protopack.Message{
					protopack.Tag{protowire.MaxValidNumber + 2, protopack.BytesType}, protopack.LengthPrefix{},
				}.Marshal(),
			)
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(protowire.MaxValidNumber + 2),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc: "MessageSet with unknown field",
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_Ext1_MessageSetExtension, &msetextpb.Ext1{
				Ext1Field1: proto.Int32(10),
			})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1000),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(10),
				}),
				protopack.Tag{4, protopack.VarintType}, protopack.Varint(0),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc:          "MessageSet with required field set",
		checkFastInit: true,
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_ExtRequired_MessageSetExtension, &msetextpb.ExtRequired{
				RequiredField1: proto.Int32(1),
			})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1002),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
					protopack.Tag{1, protopack.VarintType}, protopack.Varint(1),
				}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc:          "MessageSet with required field unset",
		checkFastInit: true,
		partial:       true,
		decodeTo: []proto.Message{func() proto.Message {
			m := &messagesetpb.MessageSetContainer{MessageSet: &messagesetpb.MessageSet{}}
			proto.SetExtension(m.MessageSet, msetextpb.E_ExtRequired_MessageSetExtension, &msetextpb.ExtRequired{})
			return m
		}()},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Varint(1002),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
}

var messageSetInvalidTestProtos = []testProto{
	{
		desc: "MessageSet with type id 0",
		decodeTo: []proto.Message{
			(*messagesetpb.MessageSetContainer)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Uvarint(0),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
	{
		desc: "MessageSet with type id overflowing int32",
		decodeTo: []proto.Message{
			(*messagesetpb.MessageSetContainer)(nil),
		},
		wire: protopack.Message{
			protopack.Tag{1, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{
				protopack.Tag{1, protopack.StartGroupType},
				protopack.Tag{2, protopack.VarintType}, protopack.Uvarint(0x80000000),
				protopack.Tag{3, protopack.BytesType}, protopack.LengthPrefix(protopack.Message{}),
				protopack.Tag{1, protopack.EndGroupType},
			}),
		}.Marshal(),
	},
}
