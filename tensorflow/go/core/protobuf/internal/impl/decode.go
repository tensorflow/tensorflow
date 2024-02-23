// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"math/bits"

	"google.golang.org/protobuf/encoding/protowire"
	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/internal/flags"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
	"google.golang.org/protobuf/runtime/protoiface"
)

var errDecode = errors.New("cannot parse invalid wire-format data")
var errRecursionDepth = errors.New("exceeded maximum recursion depth")

type unmarshalOptions struct {
	flags    protoiface.UnmarshalInputFlags
	resolver interface {
		FindExtensionByName(field protoreflect.FullName) (protoreflect.ExtensionType, error)
		FindExtensionByNumber(message protoreflect.FullName, field protoreflect.FieldNumber) (protoreflect.ExtensionType, error)
	}
	depth int
}

func (o unmarshalOptions) Options() proto.UnmarshalOptions {
	return proto.UnmarshalOptions{
		Merge:          true,
		AllowPartial:   true,
		DiscardUnknown: o.DiscardUnknown(),
		Resolver:       o.resolver,
	}
}

func (o unmarshalOptions) DiscardUnknown() bool {
	return o.flags&protoiface.UnmarshalDiscardUnknown != 0
}

func (o unmarshalOptions) IsDefault() bool {
	return o.flags == 0 && o.resolver == protoregistry.GlobalTypes
}

var lazyUnmarshalOptions = unmarshalOptions{
	resolver: protoregistry.GlobalTypes,
	depth:    protowire.DefaultRecursionLimit,
}

type unmarshalOutput struct {
	n           int // number of bytes consumed
	initialized bool
}

// unmarshal is protoreflect.Methods.Unmarshal.
func (mi *MessageInfo) unmarshal(in protoiface.UnmarshalInput) (protoiface.UnmarshalOutput, error) {
	var p pointer
	if ms, ok := in.Message.(*messageState); ok {
		p = ms.pointer()
	} else {
		p = in.Message.(*messageReflectWrapper).pointer()
	}
	out, err := mi.unmarshalPointer(in.Buf, p, 0, unmarshalOptions{
		flags:    in.Flags,
		resolver: in.Resolver,
		depth:    in.Depth,
	})
	var flags protoiface.UnmarshalOutputFlags
	if out.initialized {
		flags |= protoiface.UnmarshalInitialized
	}
	return protoiface.UnmarshalOutput{
		Flags: flags,
	}, err
}

// errUnknown is returned during unmarshaling to indicate a parse error that
// should result in a field being placed in the unknown fields section (for example,
// when the wire type doesn't match) as opposed to the entire unmarshal operation
// failing (for example, when a field extends past the available input).
//
// This is a sentinel error which should never be visible to the user.
var errUnknown = errors.New("unknown")

func (mi *MessageInfo) unmarshalPointer(b []byte, p pointer, groupTag protowire.Number, opts unmarshalOptions) (out unmarshalOutput, err error) {
	mi.init()
	opts.depth--
	if opts.depth < 0 {
		return out, errRecursionDepth
	}
	if flags.ProtoLegacy && mi.isMessageSet {
		return unmarshalMessageSet(mi, b, p, opts)
	}
	initialized := true
	var requiredMask uint64
	var exts *map[int32]ExtensionField
	start := len(b)
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
				return out, errDecode
			}
			b = b[n:]
		}
		var num protowire.Number
		if n := tag >> 3; n < uint64(protowire.MinValidNumber) || n > uint64(protowire.MaxValidNumber) {
			return out, errDecode
		} else {
			num = protowire.Number(n)
		}
		wtyp := protowire.Type(tag & 7)

		if wtyp == protowire.EndGroupType {
			if num != groupTag {
				return out, errDecode
			}
			groupTag = 0
			break
		}

		var f *coderFieldInfo
		if int(num) < len(mi.denseCoderFields) {
			f = mi.denseCoderFields[num]
		} else {
			f = mi.coderFields[num]
		}
		var n int
		err := errUnknown
		switch {
		case f != nil:
			if f.funcs.unmarshal == nil {
				break
			}
			var o unmarshalOutput
			o, err = f.funcs.unmarshal(b, p.Apply(f.offset), wtyp, f, opts)
			n = o.n
			if err != nil {
				break
			}
			requiredMask |= f.validation.requiredBit
			if f.funcs.isInit != nil && !o.initialized {
				initialized = false
			}
		default:
			// Possible extension.
			if exts == nil && mi.extensionOffset.IsValid() {
				exts = p.Apply(mi.extensionOffset).Extensions()
				if *exts == nil {
					*exts = make(map[int32]ExtensionField)
				}
			}
			if exts == nil {
				break
			}
			var o unmarshalOutput
			o, err = mi.unmarshalExtension(b, num, wtyp, *exts, opts)
			if err != nil {
				break
			}
			n = o.n
			if !o.initialized {
				initialized = false
			}
		}
		if err != nil {
			if err != errUnknown {
				return out, err
			}
			n = protowire.ConsumeFieldValue(num, wtyp, b)
			if n < 0 {
				return out, errDecode
			}
			if !opts.DiscardUnknown() && mi.unknownOffset.IsValid() {
				u := mi.mutableUnknownBytes(p)
				*u = protowire.AppendTag(*u, num, wtyp)
				*u = append(*u, b[:n]...)
			}
		}
		b = b[n:]
	}
	if groupTag != 0 {
		return out, errDecode
	}
	if mi.numRequiredFields > 0 && bits.OnesCount64(requiredMask) != int(mi.numRequiredFields) {
		initialized = false
	}
	if initialized {
		out.initialized = true
	}
	out.n = start - len(b)
	return out, nil
}

func (mi *MessageInfo) unmarshalExtension(b []byte, num protowire.Number, wtyp protowire.Type, exts map[int32]ExtensionField, opts unmarshalOptions) (out unmarshalOutput, err error) {
	x := exts[int32(num)]
	xt := x.Type()
	if xt == nil {
		var err error
		xt, err = opts.resolver.FindExtensionByNumber(mi.Desc.FullName(), num)
		if err != nil {
			if err == protoregistry.NotFound {
				return out, errUnknown
			}
			return out, errors.New("%v: unable to resolve extension %v: %v", mi.Desc.FullName(), num, err)
		}
	}
	xi := getExtensionFieldInfo(xt)
	if xi.funcs.unmarshal == nil {
		return out, errUnknown
	}
	if flags.LazyUnmarshalExtensions {
		if opts.IsDefault() && x.canLazy(xt) {
			out, valid := skipExtension(b, xi, num, wtyp, opts)
			switch valid {
			case ValidationValid:
				if out.initialized {
					x.appendLazyBytes(xt, xi, num, wtyp, b[:out.n])
					exts[int32(num)] = x
					return out, nil
				}
			case ValidationInvalid:
				return out, errDecode
			case ValidationUnknown:
			}
		}
	}
	ival := x.Value()
	if !ival.IsValid() && xi.unmarshalNeedsValue {
		// Create a new message, list, or map value to fill in.
		// For enums, create a prototype value to let the unmarshal func know the
		// concrete type.
		ival = xt.New()
	}
	v, out, err := xi.funcs.unmarshal(b, ival, num, wtyp, opts)
	if err != nil {
		return out, err
	}
	if xi.funcs.isInit == nil {
		out.initialized = true
	}
	x.Set(xt, v)
	exts[int32(num)] = x
	return out, nil
}

func skipExtension(b []byte, xi *extensionFieldInfo, num protowire.Number, wtyp protowire.Type, opts unmarshalOptions) (out unmarshalOutput, _ ValidationStatus) {
	if xi.validation.mi == nil {
		return out, ValidationUnknown
	}
	xi.validation.mi.init()
	switch xi.validation.typ {
	case validationTypeMessage:
		if wtyp != protowire.BytesType {
			return out, ValidationUnknown
		}
		v, n := protowire.ConsumeBytes(b)
		if n < 0 {
			return out, ValidationUnknown
		}
		out, st := xi.validation.mi.validate(v, 0, opts)
		out.n = n
		return out, st
	case validationTypeGroup:
		if wtyp != protowire.StartGroupType {
			return out, ValidationUnknown
		}
		out, st := xi.validation.mi.validate(b, num, opts)
		return out, st
	default:
		return out, ValidationUnknown
	}
}
