// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/reflect/protoreflect"
)

// Message is the top-level interface that all messages must implement.
// It provides access to a reflective view of a message.
// Any implementation of this interface may be used with all functions in the
// protobuf module that accept a Message, except where otherwise specified.
//
// This is the v2 interface definition for protobuf messages.
// The v1 interface definition is [github.com/golang/protobuf/proto.Message].
//
//   - To convert a v1 message to a v2 message,
//     use [google.golang.org/protobuf/protoadapt.MessageV2Of].
//   - To convert a v2 message to a v1 message,
//     use [google.golang.org/protobuf/protoadapt.MessageV1Of].
type Message = protoreflect.ProtoMessage

// Error matches all errors produced by packages in the protobuf module
// according to [errors.Is].
//
// Example usage:
//
//	if errors.Is(err, proto.Error) { ... }
var Error error

func init() {
	Error = errors.Error
}

// MessageName returns the full name of m.
// If m is nil, it returns an empty string.
func MessageName(m Message) protoreflect.FullName {
	if m == nil {
		return ""
	}
	return m.ProtoReflect().Descriptor().FullName()
}
