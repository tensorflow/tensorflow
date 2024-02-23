// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

import (
	"google.golang.org/protobuf/internal/pragma"
)

// The following types are used by the fast-path Message.ProtoMethods method.
//
// To avoid polluting the public protoreflect API with types used only by
// low-level implementations, the canonical definitions of these types are
// in the runtime/protoiface package. The definitions here and in protoiface
// must be kept in sync.
type (
	methods = struct {
		pragma.NoUnkeyedLiterals
		Flags            supportFlags
		Size             func(sizeInput) sizeOutput
		Marshal          func(marshalInput) (marshalOutput, error)
		Unmarshal        func(unmarshalInput) (unmarshalOutput, error)
		Merge            func(mergeInput) mergeOutput
		CheckInitialized func(checkInitializedInput) (checkInitializedOutput, error)
	}
	supportFlags = uint64
	sizeInput    = struct {
		pragma.NoUnkeyedLiterals
		Message Message
		Flags   uint8
	}
	sizeOutput = struct {
		pragma.NoUnkeyedLiterals
		Size int
	}
	marshalInput = struct {
		pragma.NoUnkeyedLiterals
		Message Message
		Buf     []byte
		Flags   uint8
	}
	marshalOutput = struct {
		pragma.NoUnkeyedLiterals
		Buf []byte
	}
	unmarshalInput = struct {
		pragma.NoUnkeyedLiterals
		Message  Message
		Buf      []byte
		Flags    uint8
		Resolver interface {
			FindExtensionByName(field FullName) (ExtensionType, error)
			FindExtensionByNumber(message FullName, field FieldNumber) (ExtensionType, error)
		}
		Depth int
	}
	unmarshalOutput = struct {
		pragma.NoUnkeyedLiterals
		Flags uint8
	}
	mergeInput = struct {
		pragma.NoUnkeyedLiterals
		Source      Message
		Destination Message
	}
	mergeOutput = struct {
		pragma.NoUnkeyedLiterals
		Flags uint8
	}
	checkInitializedInput = struct {
		pragma.NoUnkeyedLiterals
		Message Message
	}
	checkInitializedOutput = struct {
		pragma.NoUnkeyedLiterals
	}
)
