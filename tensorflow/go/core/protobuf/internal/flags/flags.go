// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package flags provides a set of flags controlled by build tags.
package flags

// ProtoLegacy specifies whether to enable support for legacy functionality
// such as MessageSets, weak fields, and various other obscure behavior
// that is necessary to maintain backwards compatibility with proto1 or
// the pre-release variants of proto2 and proto3.
//
// This is disabled by default unless built with the "protolegacy" tag.
//
// WARNING: The compatibility agreement covers nothing provided by this flag.
// As such, functionality may suddenly be removed or changed at our discretion.
const ProtoLegacy = protoLegacy

// LazyUnmarshalExtensions specifies whether to lazily unmarshal extensions.
//
// Lazy extension unmarshaling validates the contents of message-valued
// extension fields at unmarshal time, but defers creating the message
// structure until the extension is first accessed.
const LazyUnmarshalExtensions = ProtoLegacy
