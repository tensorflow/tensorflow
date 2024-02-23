// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"fmt"
	"math"
	"math/bits"
	"reflect"
	"unicode/utf8"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/encoding/messageset"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/internal/genid"
	"google.golang.org/protobuf/internal/strs"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
)

// ValidationStatus is the result of validating the wire-format encoding of a message.
type ValidationStatus int

const (
	// ValidationUnknown indicates that unmarshaling the message might succeed or fail.
	// The validator was unable to render a judgement.
	//
	// The only causes of this status are an aberrant message type appearing somewhere
	// in the message or a failure in the extension resolver.
	ValidationUnknown ValidationStatus = iota + 1

	// ValidationInvalid indicates that unmarshaling the message will fail.
	ValidationInvalid

	// ValidationValid indicates that unmarshaling the message will succeed.
	ValidationValid
)

func (v ValidationStatus) String() string {
	switch v {
	case ValidationUnknown:
		return "ValidationUnknown"
	case ValidationInvalid:
		return "ValidationInvalid"
	case ValidationValid:
		return "ValidationValid"
	default:
		return fmt.Sprintf("ValidationStatus(%d)", int(v))
	}
}

// Validate determines whether the contents of the buffer are a valid wire encoding
// of the message type.
//
// This function is exposed for testing.
func Validate(mt protoreflect.MessageType, in protoiface.UnmarshalInput) (out protoiface.UnmarshalOutput, _ ValidationStatus) {
	mi, ok := mt.(*MessageInfo)
	if !ok {
		return out, ValidationUnknown
	}
	if in.Resolver == nil {
		in.Resolver = protoregistry.GlobalTypes
	}
	o, st := mi.validate(in.Buf, 0, unmarshalOptions{
		flags:    in.Flags,
		resolver: in.Resolver,
	})
	if o.initialized {
		out.Flags |= protoiface.UnmarshalInitialized
	}
	return out, st
}

type validationInfo struct {
	mi               *MessageInfo
	typ              validationType
	keyType, valType validationType

	// For non-required fields, requiredBit is 0.
	//
	// For required fields, requiredBit's nth bit is set, where n is a
	// unique index in the range [0, MessageInfo.numRequiredFields).
	//
	// If there are more than 64 required fields, requiredBit is 0.
	requiredBit uint64
}

type validationType uint8

const (
	validationTypeOther validationType = iota
	validationTypeMessage
	validationTypeGroup
	validationTypeMap
	validationTypeRepeatedVarint
	validationTypeRepeatedFixed32
	validationTypeRepeatedFixed64
	validationTypeVarint
	validationTypeFixed32
	validationTypeFixed64
	validationTypeBytes
	validationTypeUTF8String
	validationTypeMessageSetItem
)

func newFieldValidationInfo(mi *MessageInfo, si structInfo, fd protoreflect.FieldDescriptor, ft reflect.Type) validationInfo {
	var vi validationInfo
	switch {
	case fd.ContainingOneof() != nil && !fd.ContainingOneof().IsSynthetic():
		switch fd.Kind() {
		case protoreflect.MessageKind:
			vi.typ = validationTypeMessage
			if ot, ok := si.oneofWrappersByNumber[fd.Number()]; ok {
				vi.mi = getMessageInfo(ot.Field(0).Type)
			}
		case protoreflect.GroupKind:
			vi.typ = validationTypeGroup
			if ot, ok := si.oneofWrappersByNumber[fd.Number()]; ok {
				vi.mi = getMessageInfo(ot.Field(0).Type)
			}
		case protoreflect.StringKind:
			if strs.EnforceUTF8(fd) {
				vi.typ = validationTypeUTF8String
			}
		}
	default:
		vi = newValidationInfo(fd, ft)
	}
	if fd.Cardinality() == protoreflect.Required {
		// Avoid overflow. The required field check is done with a 64-bit mask, with
		// any message containing more than 64 required fields always reported as
		// potentially uninitialized, so it is not important to get a precise count
		// of the required fields past 64.
		if mi.numRequiredFields < math.MaxUint8 {
			mi.numRequiredFields++
			vi.requiredBit = 1 << (mi.numRequiredFields - 1)
		}
	}
	return vi
}

func newValidationInfo(fd protoreflect.FieldDescriptor, ft reflect.Type) validationInfo {
	var vi validationInfo
	switch {
	case fd.IsList():
		switch fd.Kind() {
		case protoreflect.MessageKind:
			vi.typ = validationTypeMessage
			if ft.Kind() == reflect.Slice {
				vi.mi = getMessageInfo(ft.Elem())
			}
		case protoreflect.GroupKind:
			vi.typ = validationTypeGroup
			if ft.Kind() == reflect.Slice {
				vi.mi = getMessageInfo(ft.Elem())
			}
		case protoreflect.StringKind:
			vi.typ = validationTypeBytes
			if strs.EnforceUTF8(fd) {
				vi.typ = validationTypeUTF8String
			}
		default:
			switch wireTypes[fd.Kind()] {
			case protowire.VarintType:
				vi.typ = validationTypeRepeatedVarint
			case protowire.Fixed32Type:
				vi.typ = validationTypeRepeatedFixed32
			case protowire.Fixed64Type:
				vi.typ = validationTypeRepeatedFixed64
			}
		}
	case fd.IsMap():
		vi.typ = validationTypeMap
		switch fd.MapKey().Kind() {
		case protoreflect.StringKind:
			if strs.EnforceUTF8(fd) {
				vi.keyType = validationTypeUTF8String
			}
		}
		switch fd.MapValue().Kind() {
		case protoreflect.MessageKind:
			vi.valType = validationTypeMessage
			if ft.Kind() == reflect.Map {
				vi.mi = getMessageInfo(ft.Elem())
			}
		case protoreflect.StringKind:
			if strs.EnforceUTF8(fd) {
				vi.valType = validationTypeUTF8String
			}
		}
	default:
		switch fd.Kind() {
		case protoreflect.MessageKind:
			vi.typ = validationTypeMessage
			if !fd.IsWeak() {
				vi.mi = getMessageInfo(ft)
			}
		case protoreflect.GroupKind:
			vi.typ = validationTypeGroup
			vi.mi = getMessageInfo(ft)
		case protoreflect.StringKind:
			vi.typ = validationTypeBytes
			if strs.EnforceUTF8(fd) {
				vi.typ = validationTypeUTF8String
			}
		default:
			switch wireTypes[fd.Kind()] {
			case protowire.VarintType:
				vi.typ = validationTypeVarint
			case protowire.Fixed32Type:
				vi.typ = validationTypeFixed32
			case protowire.Fixed64Type:
				vi.typ = validationTypeFixed64
			case protowire.BytesType:
				vi.typ = validationTypeBytes
			}
		}
	}
	return vi
}

func (mi *MessageInfo) validate(b []byte, groupTag protowire.Number, opts unmarshalOptions) (out unmarshalOutput, result ValidationStatus) {
	mi.init()
	type validationState struct {
		typ              validationType
		keyType, valType validationType
		endGroup         protowire.Number
		mi               *MessageInfo
		tail             []byte
		requiredMask     uint64
	}

	// Pre-allocate some slots to avoid repeated slice reallocation.
	states := make([]validationState, 0, 16)
	states = append(states, validationState{
		typ: validationTypeMessage,
		mi:  mi,
	})
	if groupTag > 0 {
		states[0].typ = validationTypeGroup
		states[0].endGroup = groupTag
	}
	initialized := true
	start := len(b)
State:
	for len(states) > 0 {
		st := &states[len(states)-1]
		for len(b) > 0 {
			// Parse the tag (field number and wire type).
			var tag uint64
			if b[0] < 0x80 {
				tag = uint64(b[0])
				b = b[1:]
			} else if len(b) >= 2 && b[1] < 128 {
				tag = uint64(b[0]&0x7f) + uint64(b[1])<<7
				b = b[2:]
			} else {
				var n int
				tag, n = protowire.ConsumeVarint(b)
				if n < 0 {
					return out, ValidationInvalid
				}
				b = b[n:]
			}
			var num protowire.Number
			if n := tag >> 3; n < uint64(protowire.MinValidNumber) || n > uint64(protowire.MaxValidNumber) {
				return out, ValidationInvalid
			} else {
				num = protowire.Number(n)
			}
			wtyp := protowire.Type(tag & 7)

			if wtyp == protowire.EndGroupType {
				if st.endGroup == num {
					goto PopState
				}
				return out, ValidationInvalid
			}
			var vi validationInfo
			switch {
			case st.typ == validationTypeMap:
				switch num {
				case genid.MapEntry_Key_field_number:
					vi.typ = st.keyType
				case genid.MapEntry_Value_field_number:
					vi.typ = st.valType
					vi.mi = st.mi
					vi.requiredBit = 1
				}
			case flags.ProtoLegacy && st.mi.isMessageSet:
				switch num {
				case messageset.FieldItem:
					vi.typ = validationTypeMessageSetItem
				}
			default:
				var f *coderFieldInfo
				if int(num) < len(st.mi.denseCoderFields) {
					f = st.mi.denseCoderFields[num]
				} else {
					f = st.mi.coderFields[num]
				}
				if f != nil {
					vi = f.validation
					if vi.typ == validationTypeMessage && vi.mi == nil {
						// Probable weak field.
						//
						// TODO: Consider storing the results of this lookup somewhere
						// rather than recomputing it on every validation.
						fd := st.mi.Desc.Fields().ByNumber(num)
						if fd == nil || !fd.IsWeak() {
							break
						}
						messageName := fd.Message().FullName()
						messageType, err := protoregistry.GlobalTypes.FindMessageByName(messageName)
						switch err {
						case nil:
							vi.mi, _ = messageType.(*MessageInfo)
						case protoregistry.NotFound:
							vi.typ = validationTypeBytes
						default:
							return out, ValidationUnknown
						}
					}
					break
				}
				// Possible extension field.
				//
				// TODO: We should return ValidationUnknown when:
				//   1. The resolver is not frozen. (More extensions may be added to it.)
				//   2. The resolver returns preg.NotFound.
				// In this case, a type added to the resolver in the future could cause
				// unmarshaling to begin failing. Supporting this requires some way to
				// determine if the resolver is frozen.
				xt, err := opts.resolver.FindExtensionByNumber(st.mi.Desc.FullName(), num)
				if err != nil && err != protoregistry.NotFound {
					return out, ValidationUnknown
				}
				if err == nil {
					vi = getExtensionFieldInfo(xt).validation
				}
			}
			if vi.requiredBit != 0 {
				// Check that the field has a compatible wire type.
				// We only need to consider non-repeated field types,
				// since repeated fields (and maps) can never be required.
				ok := false
				switch vi.typ {
				case validationTypeVarint:
					ok = wtyp == protowire.VarintType
				case validationTypeFixed32:
					ok = wtyp == protowire.Fixed32Type
				case validationTypeFixed64:
					ok = wtyp == protowire.Fixed64Type
				case validationTypeBytes, validationTypeUTF8String, validationTypeMessage:
					ok = wtyp == protowire.BytesType
				case validationTypeGroup:
					ok = wtyp == protowire.StartGroupType
				}
				if ok {
					st.requiredMask |= vi.requiredBit
				}
			}

			switch wtyp {
			case protowire.VarintType:
				if len(b) >= 10 {
					switch {
					case b[0] < 0x80:
						b = b[1:]
					case b[1] < 0x80:
						b = b[2:]
					case b[2] < 0x80:
						b = b[3:]
					case b[3] < 0x80:
						b = b[4:]
					case b[4] < 0x80:
						b = b[5:]
					case b[5] < 0x80:
						b = b[6:]
					case b[6] < 0x80:
						b = b[7:]
					case b[7] < 0x80:
						b = b[8:]
					case b[8] < 0x80:
						b = b[9:]
					case b[9] < 0x80 && b[9] < 2:
						b = b[10:]
					default:
						return out, ValidationInvalid
					}
				} else {
					switch {
					case len(b) > 0 && b[0] < 0x80:
						b = b[1:]
					case len(b) > 1 && b[1] < 0x80:
						b = b[2:]
					case len(b) > 2 && b[2] < 0x80:
						b = b[3:]
					case len(b) > 3 && b[3] < 0x80:
						b = b[4:]
					case len(b) > 4 && b[4] < 0x80:
						b = b[5:]
					case len(b) > 5 && b[5] < 0x80:
						b = b[6:]
					case len(b) > 6 && b[6] < 0x80:
						b = b[7:]
					case len(b) > 7 && b[7] < 0x80:
						b = b[8:]
					case len(b) > 8 && b[8] < 0x80:
						b = b[9:]
					case len(b) > 9 && b[9] < 2:
						b = b[10:]
					default:
						return out, ValidationInvalid
					}
				}
				continue State
			case protowire.BytesType:
				var size uint64
				if len(b) >= 1 && b[0] < 0x80 {
					size = uint64(b[0])
					b = b[1:]
				} else if len(b) >= 2 && b[1] < 128 {
					size = uint64(b[0]&0x7f) + uint64(b[1])<<7
					b = b[2:]
				} else {
					var n int
					size, n = protowire.ConsumeVarint(b)
					if n < 0 {
						return out, ValidationInvalid
					}
					b = b[n:]
				}
				if size > uint64(len(b)) {
					return out, ValidationInvalid
				}
				v := b[:size]
				b = b[size:]
				switch vi.typ {
				case validationTypeMessage:
					if vi.mi == nil {
						return out, ValidationUnknown
					}
					vi.mi.init()
					fallthrough
				case validationTypeMap:
					if vi.mi != nil {
						vi.mi.init()
					}
					states = append(states, validationState{
						typ:     vi.typ,
						keyType: vi.keyType,
						valType: vi.valType,
						mi:      vi.mi,
						tail:    b,
					})
					b = v
					continue State
				case validationTypeRepeatedVarint:
					// Packed field.
					for len(v) > 0 {
						_, n := protowire.ConsumeVarint(v)
						if n < 0 {
							return out, ValidationInvalid
						}
						v = v[n:]
					}
				case validationTypeRepeatedFixed32:
					// Packed field.
					if len(v)%4 != 0 {
						return out, ValidationInvalid
					}
				case validationTypeRepeatedFixed64:
					// Packed field.
					if len(v)%8 != 0 {
						return out, ValidationInvalid
					}
				case validationTypeUTF8String:
					if !utf8.Valid(v) {
						return out, ValidationInvalid
					}
				}
			case protowire.Fixed32Type:
				if len(b) < 4 {
					return out, ValidationInvalid
				}
				b = b[4:]
			case protowire.Fixed64Type:
				if len(b) < 8 {
					return out, ValidationInvalid
				}
				b = b[8:]
			case protowire.StartGroupType:
				switch {
				case vi.typ == validationTypeGroup:
					if vi.mi == nil {
						return out, ValidationUnknown
					}
					vi.mi.init()
					states = append(states, validationState{
						typ:      validationTypeGroup,
						mi:       vi.mi,
						endGroup: num,
					})
					continue State
				case flags.ProtoLegacy && vi.typ == validationTypeMessageSetItem:
					typeid, v, n, err := messageset.ConsumeFieldValue(b, false)
					if err != nil {
						return out, ValidationInvalid
					}
					xt, err := opts.resolver.FindExtensionByNumber(st.mi.Desc.FullName(), typeid)
					switch {
					case err == protoregistry.NotFound:
						b = b[n:]
					case err != nil:
						return out, ValidationUnknown
					default:
						xvi := getExtensionFieldInfo(xt).validation
						if xvi.mi != nil {
							xvi.mi.init()
						}
						states = append(states, validationState{
							typ:  xvi.typ,
							mi:   xvi.mi,
							tail: b[n:],
						})
						b = v
						continue State
					}
				default:
					n := protowire.ConsumeFieldValue(num, wtyp, b)
					if n < 0 {
						return out, ValidationInvalid
					}
					b = b[n:]
				}
			default:
				return out, ValidationInvalid
			}
		}
		if st.endGroup != 0 {
			return out, ValidationInvalid
		}
		if len(b) != 0 {
			return out, ValidationInvalid
		}
		b = st.tail
	PopState:
		numRequiredFields := 0
		switch st.typ {
		case validationTypeMessage, validationTypeGroup:
			numRequiredFields = int(st.mi.numRequiredFields)
		case validationTypeMap:
			// If this is a map field with a message value that contains
			// required fields, require that the value be present.
			if st.mi != nil && st.mi.numRequiredFields > 0 {
				numRequiredFields = 1
			}
		}
		// If there are more than 64 required fields, this check will
		// always fail and we will report that the message is potentially
		// uninitialized.
		if numRequiredFields > 0 && bits.OnesCount64(st.requiredMask) != numRequiredFields {
			initialized = false
		}
		states = states[:len(states)-1]
	}
	out.n = start - len(b)
	if initialized {
		out.initialized = true
	}
	return out, ValidationValid
}
