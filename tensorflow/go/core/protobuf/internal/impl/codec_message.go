// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"reflect"
	"sort"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/order"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

// coderMessageInfo contains per-message information used by the fast-path functions.
// This is a different type from MessageInfo to keep MessageInfo as general-purpose as
// possible.
type coderMessageInfo struct {
	methods protoiface.Methods

	orderedCoderFields []*coderFieldInfo
	denseCoderFields   []*coderFieldInfo
	coderFields        map[protowire.Number]*coderFieldInfo
	sizecacheOffset    offset
	unknownOffset      offset
	unknownPtrKind     bool
	extensionOffset    offset
	needsInitCheck     bool
	isMessageSet       bool
	numRequiredFields  uint8
}

type coderFieldInfo struct {
	funcs      pointerCoderFuncs // fast-path per-field functions
	mi         *MessageInfo      // field's message
	ft         reflect.Type
	validation validationInfo           // information used by message validation
	num        protoreflect.FieldNumber // field number
	offset     offset                   // struct field offset
	wiretag    uint64                   // field tag (number + wire type)
	tagsize    int                      // size of the varint-encoded tag
	isPointer  bool                     // true if IsNil may be called on the struct field
	isRequired bool                     // true if field is required
}

func (mi *MessageInfo) makeCoderMethods(t reflect.Type, si structInfo) {
	mi.sizecacheOffset = invalidOffset
	mi.unknownOffset = invalidOffset
	mi.extensionOffset = invalidOffset

	if si.sizecacheOffset.IsValid() && si.sizecacheType == sizecacheType {
		mi.sizecacheOffset = si.sizecacheOffset
	}
	if si.unknownOffset.IsValid() && (si.unknownType == unknownFieldsAType || si.unknownType == unknownFieldsBType) {
		mi.unknownOffset = si.unknownOffset
		mi.unknownPtrKind = si.unknownType.Kind() == reflect.Ptr
	}
	if si.extensionOffset.IsValid() && si.extensionType == extensionFieldsType {
		mi.extensionOffset = si.extensionOffset
	}

	mi.coderFields = make(map[protowire.Number]*coderFieldInfo)
	fields := mi.Desc.Fields()
	preallocFields := make([]coderFieldInfo, fields.Len())
	for i := 0; i < fields.Len(); i++ {
		fd := fields.Get(i)

		fs := si.fieldsByNumber[fd.Number()]
		isOneof := fd.ContainingOneof() != nil && !fd.ContainingOneof().IsSynthetic()
		if isOneof {
			fs = si.oneofsByName[fd.ContainingOneof().Name()]
		}
		ft := fs.Type
		var wiretag uint64
		if !fd.IsPacked() {
			wiretag = protowire.EncodeTag(fd.Number(), wireTypes[fd.Kind()])
		} else {
			wiretag = protowire.EncodeTag(fd.Number(), protowire.BytesType)
		}
		var fieldOffset offset
		var funcs pointerCoderFuncs
		var childMessage *MessageInfo
		switch {
		case ft == nil:
			// This never occurs for generated message types.
			// It implies that a hand-crafted type has missing Go fields
			// for specific protobuf message fields.
			funcs = pointerCoderFuncs{
				size: func(p pointer, f *coderFieldInfo, opts marshalOptions) int {
					return 0
				},
				marshal: func(b []byte, p pointer, f *coderFieldInfo, opts marshalOptions) ([]byte, error) {
					return nil, nil
				},
				unmarshal: func(b []byte, p pointer, wtyp protowire.Type, f *coderFieldInfo, opts unmarshalOptions) (unmarshalOutput, error) {
					panic("missing Go struct field for " + string(fd.FullName()))
				},
				isInit: func(p pointer, f *coderFieldInfo) error {
					panic("missing Go struct field for " + string(fd.FullName()))
				},
				merge: func(dst, src pointer, f *coderFieldInfo, opts mergeOptions) {
					panic("missing Go struct field for " + string(fd.FullName()))
				},
			}
		case isOneof:
			fieldOffset = offsetOf(fs, mi.Exporter)
		case fd.IsWeak():
			fieldOffset = si.weakOffset
			funcs = makeWeakMessageFieldCoder(fd)
		default:
			fieldOffset = offsetOf(fs, mi.Exporter)
			childMessage, funcs = fieldCoder(fd, ft)
		}
		cf := &preallocFields[i]
		*cf = coderFieldInfo{
			num:        fd.Number(),
			offset:     fieldOffset,
			wiretag:    wiretag,
			ft:         ft,
			tagsize:    protowire.SizeVarint(wiretag),
			funcs:      funcs,
			mi:         childMessage,
			validation: newFieldValidationInfo(mi, si, fd, ft),
			isPointer:  fd.Cardinality() == protoreflect.Repeated || fd.HasPresence(),
			isRequired: fd.Cardinality() == protoreflect.Required,
		}
		mi.orderedCoderFields = append(mi.orderedCoderFields, cf)
		mi.coderFields[cf.num] = cf
	}
	for i, oneofs := 0, mi.Desc.Oneofs(); i < oneofs.Len(); i++ {
		if od := oneofs.Get(i); !od.IsSynthetic() {
			mi.initOneofFieldCoders(od, si)
		}
	}
	if messageset.IsMessageSet(mi.Desc) {
		if !mi.extensionOffset.IsValid() {
			panic(fmt.Sprintf("%v: MessageSet with no extensions field", mi.Desc.FullName()))
		}
		if !mi.unknownOffset.IsValid() {
			panic(fmt.Sprintf("%v: MessageSet with no unknown field", mi.Desc.FullName()))
		}
		mi.isMessageSet = true
	}
	sort.Slice(mi.orderedCoderFields, func(i, j int) bool {
		return mi.orderedCoderFields[i].num < mi.orderedCoderFields[j].num
	})

	var maxDense protoreflect.FieldNumber
	for _, cf := range mi.orderedCoderFields {
		if cf.num >= 16 && cf.num >= 2*maxDense {
			break
		}
		maxDense = cf.num
	}
	mi.denseCoderFields = make([]*coderFieldInfo, maxDense+1)
	for _, cf := range mi.orderedCoderFields {
		if int(cf.num) >= len(mi.denseCoderFields) {
			break
		}
		mi.denseCoderFields[cf.num] = cf
	}

	// To preserve compatibility with historic wire output, marshal oneofs last.
	if mi.Desc.Oneofs().Len() > 0 {
		sort.Slice(mi.orderedCoderFields, func(i, j int) bool {
			fi := fields.ByNumber(mi.orderedCoderFields[i].num)
			fj := fields.ByNumber(mi.orderedCoderFields[j].num)
			return order.LegacyFieldOrder(fi, fj)
		})
	}

	mi.needsInitCheck = needsInitCheck(mi.Desc)
	if mi.methods.Marshal == nil && mi.methods.Size == nil {
		mi.methods.Flags |= protoiface.SupportMarshalDeterministic
		mi.methods.Marshal = mi.marshal
		mi.methods.Size = mi.size
	}
	if mi.methods.Unmarshal == nil {
		mi.methods.Flags |= protoiface.SupportUnmarshalDiscardUnknown
		mi.methods.Unmarshal = mi.unmarshal
	}
	if mi.methods.CheckInitialized == nil {
		mi.methods.CheckInitialized = mi.checkInitialized
	}
	if mi.methods.Merge == nil {
		mi.methods.Merge = mi.merge
	}
}

// getUnknownBytes returns a *[]byte for the unknown fields.
// It is the caller's responsibility to check whether the pointer is nil.
// This function is specially designed to be inlineable.
func (mi *MessageInfo) getUnknownBytes(p pointer) *[]byte {
	if mi.unknownPtrKind {
		return *p.Apply(mi.unknownOffset).BytesPtr()
	} else {
		return p.Apply(mi.unknownOffset).Bytes()
	}
}

// mutableUnknownBytes returns a *[]byte for the unknown fields.
// The returned pointer is guaranteed to not be nil.
func (mi *MessageInfo) mutableUnknownBytes(p pointer) *[]byte {
	if mi.unknownPtrKind {
		bp := p.Apply(mi.unknownOffset).BytesPtr()
		if *bp == nil {
			*bp = new([]byte)
		}
		return *bp
	} else {
		return p.Apply(mi.unknownOffset).Bytes()
	}
}
