// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The protoreflect tag disables fast-path methods, including legacy ones.
//go:build !protoreflect
// +build !protoreflect

package proto_test

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"google.golang.org/protobuf/internal/impl"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/runtime/protoiface"

	legacypb "google.golang.org/protobuf/internal/testprotos/legacy"
)

type selfMarshaler struct {
	bytes []byte
	err   error
}

func (m selfMarshaler) Reset()        {}
func (m selfMarshaler) ProtoMessage() {}

func (m selfMarshaler) String() string {
	return fmt.Sprintf("selfMarshaler{bytes:%v, err:%v}", m.bytes, m.err)
}

func (m selfMarshaler) Marshal() ([]byte, error) {
	return m.bytes, m.err
}

func (m *selfMarshaler) Unmarshal(b []byte) error {
	m.bytes = b
	return m.err
}

func TestLegacyMarshalMethod(t *testing.T) {
	for _, test := range []selfMarshaler{
		{bytes: []byte("marshal")},
		{bytes: []byte("marshal"), err: errors.New("some error")},
	} {
		m := impl.Export{}.MessageOf(test).Interface()
		b, err := proto.Marshal(m)
		if err != test.err || !bytes.Equal(b, test.bytes) {
			t.Errorf("proto.Marshal(%v) = %v, %v; want %v, %v", test, b, err, test.bytes, test.err)
		}
		if gotSize, wantSize := proto.Size(m), len(test.bytes); gotSize != wantSize {
			t.Fatalf("proto.Size(%v) = %v, want %v", test, gotSize, wantSize)
		}

		prefix := []byte("prefix")
		want := append(prefix, test.bytes...)
		b, err = proto.MarshalOptions{}.MarshalAppend(prefix, m)
		if err != test.err || !bytes.Equal(b, want) {
			t.Errorf("MarshalAppend(%v, %v) = %v, %v; want %v, %v", prefix, test, b, err, test.bytes, test.err)
		}

		b, err = proto.MarshalOptions{
			Deterministic: true,
		}.MarshalAppend(nil, m)
		if err != test.err || !bytes.Equal(b, test.bytes) {
			t.Errorf("MarshalOptions{Deterministic:true}.MarshalAppend(nil, %v) = %v, %v; want %v, %v", test, b, err, test.bytes, test.err)
		}
	}
}

func TestLegacyUnmarshalMethod(t *testing.T) {
	sm := &selfMarshaler{}
	m := impl.Export{}.MessageOf(sm).Interface()
	want := []byte("unmarshal")
	if err := proto.Unmarshal(want, m); err != nil {
		t.Fatalf("proto.Unmarshal(selfMarshaler{}) = %v, want nil", err)
	}
	if !bytes.Equal(sm.bytes, want) {
		t.Fatalf("proto.Unmarshal(selfMarshaler{}): Marshal method not called")
	}
}

type descPanicSelfMarshaler struct{}

const descPanicSelfMarshalerBytes = "bytes"

func (m *descPanicSelfMarshaler) Reset()                      {}
func (m *descPanicSelfMarshaler) ProtoMessage()               {}
func (m *descPanicSelfMarshaler) Descriptor() ([]byte, []int) { panic("Descriptor method panics") }
func (m *descPanicSelfMarshaler) String() string              { return "descPanicSelfMarshaler{}" }
func (m *descPanicSelfMarshaler) Marshal() ([]byte, error) {
	return []byte(descPanicSelfMarshalerBytes), nil
}

func TestSelfMarshalerDescriptorPanics(t *testing.T) {
	m := &descPanicSelfMarshaler{}
	got, err := proto.Marshal(impl.Export{}.MessageOf(m).Interface())
	want := []byte(descPanicSelfMarshalerBytes)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("proto.Marshal(%v) = %v, %v; want %v, nil", m, got, err, want)
	}
}

type descSelfMarshaler struct {
	someField int // some non-generated field
}

const descSelfMarshalerBytes = "bytes"

func (m *descSelfMarshaler) Reset()        {}
func (m *descSelfMarshaler) ProtoMessage() {}
func (m *descSelfMarshaler) Descriptor() ([]byte, []int) {
	return ((*legacypb.Legacy)(nil)).GetF1().Descriptor()
}
func (m *descSelfMarshaler) String() string {
	return "descSelfMarshaler{}"
}
func (m *descSelfMarshaler) Marshal() ([]byte, error) {
	return []byte(descSelfMarshalerBytes), nil
}

func TestSelfMarshalerWithDescriptor(t *testing.T) {
	m := &descSelfMarshaler{}
	got, err := proto.Marshal(impl.Export{}.MessageOf(m).Interface())
	want := []byte(descSelfMarshalerBytes)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("proto.Marshal(%v) = %v, %v; want %v, nil", m, got, err, want)
	}
}

func TestDecodeFastCheckInitialized(t *testing.T) {
	for _, test := range testValidMessages {
		if !test.checkFastInit {
			continue
		}
		for _, message := range test.decodeTo {
			t.Run(fmt.Sprintf("%s (%T)", test.desc, message), func(t *testing.T) {
				m := message.ProtoReflect().New()
				opts := proto.UnmarshalOptions{
					AllowPartial: true,
				}
				out, err := opts.UnmarshalState(protoiface.UnmarshalInput{
					Buf:     test.wire,
					Message: m,
				})
				if err != nil {
					t.Fatalf("Unmarshal error: %v", err)
				}
				if got, want := (out.Flags&protoiface.UnmarshalInitialized != 0), !test.partial; got != want {
					t.Errorf("out.Initialized = %v, want %v", got, want)
				}
			})
		}
	}
}

type selfMerger struct {
	src protoiface.MessageV1
}

func (*selfMerger) Reset()         {}
func (*selfMerger) ProtoMessage()  {}
func (*selfMerger) String() string { return "selfMerger{}" }
func (m *selfMerger) Merge(src protoiface.MessageV1) {
	m.src = src
}

func TestLegacyMergeMethod(t *testing.T) {
	src := &selfMerger{}
	dst := &selfMerger{}
	proto.Merge(
		impl.Export{}.MessageOf(dst).Interface(),
		impl.Export{}.MessageOf(src).Interface(),
	)
	if got, want := dst.src, src; got != want {
		t.Errorf("Merge(dst, src): want dst.src = src, got %v", got)
	}
	if got := src.src; got != nil {
		t.Errorf("Merge(dst, src): want src.src = nil, got %v", got)
	}
}
