// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protodelim_test

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/testprotos/test3"
	"google.golang.org/protobuf/testing/protocmp"
)

func TestRoundTrip(t *testing.T) {
	msgs := []*test3.TestAllTypes{
		{SingularInt32: 1},
		{SingularString: "hello"},
		{RepeatedDouble: []float64{1.2, 3.4}},
		{
			SingularNestedMessage:  &test3.TestAllTypes_NestedMessage{A: 1},
			RepeatedForeignMessage: []*test3.ForeignMessage{{C: 2}, {D: 3}},
		},
	}

	buf := &bytes.Buffer{}

	// Write all messages to buf.
	for _, m := range msgs {
		if n, err := protodelim.MarshalTo(buf, m); err != nil {
			t.Errorf("protodelim.MarshalTo(_, %v) = %d, %v", m, n, err)
		}
	}

	for _, tc := range []struct {
		name   string
		reader protodelim.Reader
	}{
		{name: "defaultbuffer", reader: bufio.NewReader(bytes.NewBuffer(buf.Bytes()))},
		{name: "smallbuffer", reader: bufio.NewReaderSize(bytes.NewBuffer(buf.Bytes()), 0)},
		{name: "largebuffer", reader: bufio.NewReaderSize(bytes.NewBuffer(buf.Bytes()), 1<<20)},
		{name: "notbufio", reader: notBufioReader{bufio.NewReader(bytes.NewBuffer(buf.Bytes()))}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Read and collect messages from buf.
			var got []*test3.TestAllTypes
			for {
				m := &test3.TestAllTypes{}
				err := protodelim.UnmarshalFrom(tc.reader, m)
				if errors.Is(err, io.EOF) {
					break
				}
				if err != nil {
					t.Errorf("protodelim.UnmarshalFrom(_) = %v", err)
					continue
				}
				got = append(got, m)
			}

			want := msgs
			if diff := cmp.Diff(want, got, protocmp.Transform()); diff != "" {
				t.Errorf("Unmarshaler collected messages: diff -want +got = %s", diff)
			}
		})
	}
}

// Just a wrapper so that UnmarshalFrom doesn't recognize this as a bufio.Reader
type notBufioReader struct {
	*bufio.Reader
}

func BenchmarkUnmarshalFrom(b *testing.B) {
	var manyInt32 []int32
	for i := int32(0); i < 10000; i++ {
		manyInt32 = append(manyInt32, i)
	}
	var msgs []*test3.TestAllTypes
	for i := 0; i < 10; i++ {
		msgs = append(msgs, &test3.TestAllTypes{RepeatedInt32: manyInt32})
	}

	buf := &bytes.Buffer{}

	// Write all messages to buf.
	for _, m := range msgs {
		if n, err := protodelim.MarshalTo(buf, m); err != nil {
			b.Errorf("protodelim.MarshalTo(_, %v) = %d, %v", m, n, err)
		}
	}
	bufBytes := buf.Bytes()

	type resetReader interface {
		protodelim.Reader
		Reset(io.Reader)
	}

	for _, tc := range []struct {
		name   string
		reader resetReader
	}{
		{name: "bufio1mib", reader: bufio.NewReaderSize(nil, 1<<20)},
		{name: "bufio16mib", reader: bufio.NewReaderSize(nil, 1<<24)},
		{name: "notbufio1mib", reader: notBufioReader{bufio.NewReaderSize(nil, 1<<20)}},
		{name: "notbufio16mib", reader: notBufioReader{bufio.NewReaderSize(nil, 1<<24)}},
	} {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tc.reader.Reset(bytes.NewBuffer(bufBytes))
				var got int
				m := &test3.TestAllTypes{}
				for {
					err := protodelim.UnmarshalFrom(tc.reader, m)
					if errors.Is(err, io.EOF) {
						break
					}
					if err != nil {
						b.Errorf("protodelim.UnmarshalFrom(_) = %v", err)
						continue
					}
					got++
				}
				if got != len(msgs) {
					b.Errorf("Got %v messages. Wanted %v", got, len(msgs))
				}
			}
		})
	}
}

func TestMaxSize(t *testing.T) {
	in := &test3.TestAllTypes{SingularInt32: 1}

	buf := &bytes.Buffer{}

	if n, err := protodelim.MarshalTo(buf, in); err != nil {
		t.Errorf("protodelim.MarshalTo(_, %v) = %d, %v", in, n, err)
	}

	out := &test3.TestAllTypes{}
	err := protodelim.UnmarshalOptions{MaxSize: 1}.UnmarshalFrom(bufio.NewReader(buf), out)

	var errSize *protodelim.SizeTooLargeError
	if !errors.As(err, &errSize) {
		t.Errorf("protodelim.UnmarshalOptions{MaxSize: 1}.UnmarshalFrom(_, _) = %v (%T), want %T", err, err, errSize)
	}
	got, want := errSize, &protodelim.SizeTooLargeError{Size: 3, MaxSize: 1}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("protodelim.UnmarshalOptions{MaxSize: 1}.UnmarshalFrom(_, _): diff -want +got = %s", diff)
	}
}

func TestUnmarshalFrom_UnexpectedEOF(t *testing.T) {
	buf := &bytes.Buffer{}

	// Write a size (42), but no subsequent message.
	sb := protowire.AppendVarint(nil, 42)
	if _, err := buf.Write(sb); err != nil {
		t.Fatalf("buf.Write(%v) = _, %v", sb, err)
	}

	out := &test3.TestAllTypes{}
	err := protodelim.UnmarshalFrom(bufio.NewReader(buf), out)
	if got, want := err, io.ErrUnexpectedEOF; got != want {
		t.Errorf("protodelim.UnmarshalFrom(size-only buf, _) = %v, want %v", got, want)
	}
}

func TestUnmarshalFrom_PrematureHeader(t *testing.T) {
	var data = []byte{128} // continuation bit set
	err := protodelim.UnmarshalFrom(bytes.NewReader(data[:]), nil)
	if got, want := err, io.ErrUnexpectedEOF; !errors.Is(got, want) {
		t.Errorf("protodelim.UnmarshalFrom(%#v, nil) = %#v; want = %#v", data, got, want)
	}
}

func TestUnmarshalFrom_InvalidVarint(t *testing.T) {
	var data = bytes.Repeat([]byte{128}, 2*binary.MaxVarintLen64) // continuation bit set
	err := protodelim.UnmarshalFrom(bytes.NewReader(data[:]), nil)
	if err == nil {
		t.Errorf("protodelim.UnmarshalFrom unexpectedly did not error on invalid varint")
	}
}
