// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proto

import (
	"fmt"

	"google.golang.org/protobuf/reflect/protoreflect"
)

// Reset clears every field in the message.
// The resulting message shares no observable memory with its previous state
// other than the memory for the message itself.
func Reset(m Message) {
	if mr, ok := m.(interface{ Reset() }); ok && hasProtoMethods {
		mr.Reset()
		return
	}
	resetMessage(m.ProtoReflect())
}

func resetMessage(m protoreflect.Message) {
	if !m.IsValid() {
		panic(fmt.Sprintf("cannot reset invalid %v message", m.Descriptor().FullName()))
	}

	// Clear all known fields.
	fds := m.Descriptor().Fields()
	for i := 0; i < fds.Len(); i++ {
		m.Clear(fds.Get(i))
	}

	// Clear extension fields.
	m.Range(func(fd protoreflect.FieldDescriptor, _ protoreflect.Value) bool {
		m.Clear(fd)
		return true
	})

	// Clear unknown fields.
	m.SetUnknown(nil)
}
