// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dynamicpb

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"

	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

type extField struct {
	name   protoreflect.FullName
	number protoreflect.FieldNumber
}

// A Types is a collection of dynamically constructed descriptors.
// Its methods are safe for concurrent use.
//
// Types implements [protoregistry.MessageTypeResolver] and [protoregistry.ExtensionTypeResolver].
// A Types may be used as a [google.golang.org/protobuf/proto.UnmarshalOptions.Resolver].
type Types struct {
	// atomicExtFiles is used with sync/atomic and hence must be the first word
	// of the struct to guarantee 64-bit alignment.
	//
	// TODO(stapelberg): once we only support Go 1.19 and newer, switch this
	// field to be of type atomic.Uint64 to guarantee alignment on
	// stack-allocated values, too.
	atomicExtFiles uint64
	extMu          sync.Mutex

	files *protoregistry.Files

	extensionsByMessage map[extField]protoreflect.ExtensionDescriptor
}

// NewTypes creates a new Types registry with the provided files.
// The Files registry is retained, and changes to Files will be reflected in Types.
// It is not safe to concurrently change the Files while calling Types methods.
func NewTypes(f *protoregistry.Files) *Types {
	return &Types{
		files: f,
	}
}

// FindEnumByName looks up an enum by its full name;
// e.g., "google.protobuf.Field.Kind".
//
// This returns (nil, [protoregistry.NotFound]) if not found.
func (t *Types) FindEnumByName(name protoreflect.FullName) (protoreflect.EnumType, error) {
	d, err := t.files.FindDescriptorByName(name)
	if err != nil {
		return nil, err
	}
	ed, ok := d.(protoreflect.EnumDescriptor)
	if !ok {
		return nil, errors.New("found wrong type: got %v, want enum", descName(d))
	}
	return NewEnumType(ed), nil
}

// FindExtensionByName looks up an extension field by the field's full name.
// Note that this is the full name of the field as determined by
// where the extension is declared and is unrelated to the full name of the
// message being extended.
//
// This returns (nil, [protoregistry.NotFound]) if not found.
func (t *Types) FindExtensionByName(name protoreflect.FullName) (protoreflect.ExtensionType, error) {
	d, err := t.files.FindDescriptorByName(name)
	if err != nil {
		return nil, err
	}
	xd, ok := d.(protoreflect.ExtensionDescriptor)
	if !ok {
		return nil, errors.New("found wrong type: got %v, want extension", descName(d))
	}
	return NewExtensionType(xd), nil
}

// FindExtensionByNumber looks up an extension field by the field number
// within some parent message, identified by full name.
//
// This returns (nil, [protoregistry.NotFound]) if not found.
func (t *Types) FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error) {
	// Construct the extension number map lazily, since not every user will need it.
	// Update the map if new files are added to the registry.
	if atomic.LoadUint64(&t.atomicExtFiles) != uint64(t.files.NumFiles()) {
		t.updateExtensions()
	}
	xd := t.extensionsByMessage[extField{message, field}]
	if xd == nil {
		return nil, protoregistry.NotFound
	}
	return NewExtensionType(xd), nil
}

// FindMessageByName looks up a message by its full name;
// e.g. "google.protobuf.Any".
//
// This returns (nil, [protoregistry.NotFound]) if not found.
func (t *Types) FindMessageByName(name protoreflect.FullName) (protoreflect.MessageType, error) {
	d, err := t.files.FindDescriptorByName(name)
	if err != nil {
		return nil, err
	}
	md, ok := d.(protoreflect.MessageDescriptor)
	if !ok {
		return nil, errors.New("found wrong type: got %v, want message", descName(d))
	}
	return NewMessageType(md), nil
}

// FindMessageByURL looks up a message by a URL identifier.
// See documentation on google.protobuf.Any.type_url for the URL format.
//
// This returns (nil, [protoregistry.NotFound]) if not found.
func (t *Types) FindMessageByURL(url string) (protoreflect.MessageType, error) {
	// This function is similar to FindMessageByName but
	// truncates anything before and including '/' in the URL.
	message := protoreflect.FullName(url)
	if i := strings.LastIndexByte(url, '/'); i >= 0 {
		message = message[i+len("/"):]
	}
	return t.FindMessageByName(message)
}

func (t *Types) updateExtensions() {
	t.extMu.Lock()
	defer t.extMu.Unlock()
	if atomic.LoadUint64(&t.atomicExtFiles) == uint64(t.files.NumFiles()) {
		return
	}
	defer atomic.StoreUint64(&t.atomicExtFiles, uint64(t.files.NumFiles()))
	t.files.RangeFiles(func(fd protoreflect.FileDescriptor) bool {
		t.registerExtensions(fd.Extensions())
		t.registerExtensionsInMessages(fd.Messages())
		return true
	})
}

func (t *Types) registerExtensionsInMessages(mds protoreflect.MessageDescriptors) {
	count := mds.Len()
	for i := 0; i < count; i++ {
		md := mds.Get(i)
		t.registerExtensions(md.Extensions())
		t.registerExtensionsInMessages(md.Messages())
	}
}

func (t *Types) registerExtensions(xds protoreflect.ExtensionDescriptors) {
	count := xds.Len()
	for i := 0; i < count; i++ {
		xd := xds.Get(i)
		field := xd.Number()
		message := xd.ContainingMessage().FullName()
		if t.extensionsByMessage == nil {
			t.extensionsByMessage = make(map[extField]protoreflect.ExtensionDescriptor)
		}
		t.extensionsByMessage[extField{message, field}] = xd
	}
}

func descName(d protoreflect.Descriptor) string {
	switch d.(type) {
	case protoreflect.EnumDescriptor:
		return "enum"
	case protoreflect.EnumValueDescriptor:
		return "enum value"
	case protoreflect.MessageDescriptor:
		return "message"
	case protoreflect.ExtensionDescriptor:
		return "extension"
	case protoreflect.ServiceDescriptor:
		return "service"
	default:
		return fmt.Sprintf("%T", d)
	}
}
