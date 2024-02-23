// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package protodelim marshals and unmarshals varint size-delimited messages.
package protodelim

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/proto"
)

// MarshalOptions is a configurable varint size-delimited marshaler.
type MarshalOptions struct{ proto.MarshalOptions }

// MarshalTo writes a varint size-delimited wire-format message to w.
// If w returns an error, MarshalTo returns it unchanged.
func (o MarshalOptions) MarshalTo(w io.Writer, m proto.Message) (int, error) {
	msgBytes, err := o.MarshalOptions.Marshal(m)
	if err != nil {
		return 0, err
	}

	sizeBytes := protowire.AppendVarint(nil, uint64(len(msgBytes)))
	sizeWritten, err := w.Write(sizeBytes)
	if err != nil {
		return sizeWritten, err
	}
	msgWritten, err := w.Write(msgBytes)
	if err != nil {
		return sizeWritten + msgWritten, err
	}
	return sizeWritten + msgWritten, nil
}

// MarshalTo writes a varint size-delimited wire-format message to w
// with the default options.
//
// See the documentation for [MarshalOptions.MarshalTo].
func MarshalTo(w io.Writer, m proto.Message) (int, error) {
	return MarshalOptions{}.MarshalTo(w, m)
}

// UnmarshalOptions is a configurable varint size-delimited unmarshaler.
type UnmarshalOptions struct {
	proto.UnmarshalOptions

	// MaxSize is the maximum size in wire-format bytes of a single message.
	// Unmarshaling a message larger than MaxSize will return an error.
	// A zero MaxSize will default to 4 MiB.
	// Setting MaxSize to -1 disables the limit.
	MaxSize int64
}

const defaultMaxSize = 4 << 20 // 4 MiB, corresponds to the default gRPC max request/response size

// SizeTooLargeError is an error that is returned when the unmarshaler encounters a message size
// that is larger than its configured [UnmarshalOptions.MaxSize].
type SizeTooLargeError struct {
	// Size is the varint size of the message encountered
	// that was larger than the provided MaxSize.
	Size uint64

	// MaxSize is the MaxSize limit configured in UnmarshalOptions, which Size exceeded.
	MaxSize uint64
}

func (e *SizeTooLargeError) Error() string {
	return fmt.Sprintf("message size %d exceeded unmarshaler's maximum configured size %d", e.Size, e.MaxSize)
}

// Reader is the interface expected by [UnmarshalFrom].
// It is implemented by *[bufio.Reader].
type Reader interface {
	io.Reader
	io.ByteReader
}

// UnmarshalFrom parses and consumes a varint size-delimited wire-format message
// from r.
// The provided message must be mutable (e.g., a non-nil pointer to a message).
//
// The error is [io.EOF] error only if no bytes are read.
// If an EOF happens after reading some but not all the bytes,
// UnmarshalFrom returns a non-io.EOF error.
// In particular if r returns a non-io.EOF error, UnmarshalFrom returns it unchanged,
// and if only a size is read with no subsequent message, [io.ErrUnexpectedEOF] is returned.
func (o UnmarshalOptions) UnmarshalFrom(r Reader, m proto.Message) error {
	var sizeArr [binary.MaxVarintLen64]byte
	sizeBuf := sizeArr[:0]
	for i := range sizeArr {
		b, err := r.ReadByte()
		if err != nil {
			// Immediate EOF is unexpected.
			if err == io.EOF && i != 0 {
				break
			}
			return err
		}
		sizeBuf = append(sizeBuf, b)
		if b < 0x80 {
			break
		}
	}
	size, n := protowire.ConsumeVarint(sizeBuf)
	if n < 0 {
		return protowire.ParseError(n)
	}

	maxSize := o.MaxSize
	if maxSize == 0 {
		maxSize = defaultMaxSize
	}
	if maxSize != -1 && size > uint64(maxSize) {
		return errors.Wrap(&SizeTooLargeError{Size: size, MaxSize: uint64(maxSize)}, "")
	}

	var b []byte
	var err error
	if br, ok := r.(*bufio.Reader); ok {
		// Use the []byte from the bufio.Reader instead of having to allocate one.
		// This reduces CPU usage and allocated bytes.
		b, err = br.Peek(int(size))
		if err == nil {
			defer br.Discard(int(size))
		} else {
			b = nil
		}
	}
	if b == nil {
		b = make([]byte, size)
		_, err = io.ReadFull(r, b)
	}

	if err == io.EOF {
		return io.ErrUnexpectedEOF
	}
	if err != nil {
		return err
	}
	if err := o.Unmarshal(b, m); err != nil {
		return err
	}
	return nil
}

// UnmarshalFrom parses and consumes a varint size-delimited wire-format message
// from r with the default options.
// The provided message must be mutable (e.g., a non-nil pointer to a message).
//
// See the documentation for [UnmarshalOptions.UnmarshalFrom].
func UnmarshalFrom(r Reader, m proto.Message) error {
	return UnmarshalOptions{}.UnmarshalFrom(r, m)
}
