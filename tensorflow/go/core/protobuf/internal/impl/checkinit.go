// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package impl

import (
	"sync"

	"google.golang.org/protobuf/internal/errors"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/runtime/protoiface"
)

func (mi *MessageInfo) checkInitialized(in protoiface.CheckInitializedInput) (protoiface.CheckInitializedOutput, error) {
	var p pointer
	if ms, ok := in.Message.(*messageState); ok {
		p = ms.pointer()
	} else {
		p = in.Message.(*messageReflectWrapper).pointer()
	}
	return protoiface.CheckInitializedOutput{}, mi.checkInitializedPointer(p)
}

func (mi *MessageInfo) checkInitializedPointer(p pointer) error {
	mi.init()
	if !mi.needsInitCheck {
		return nil
	}
	if p.IsNil() {
		for _, f := range mi.orderedCoderFields {
			if f.isRequired {
				return errors.RequiredNotSet(string(mi.Desc.Fields().ByNumber(f.num).FullName()))
			}
		}
		return nil
	}
	if mi.extensionOffset.IsValid() {
		e := p.Apply(mi.extensionOffset).Extensions()
		if err := mi.isInitExtensions(e); err != nil {
			return err
		}
	}
	for _, f := range mi.orderedCoderFields {
		if !f.isRequired && f.funcs.isInit == nil {
			continue
		}
		fptr := p.Apply(f.offset)
		if f.isPointer && fptr.Elem().IsNil() {
			if f.isRequired {
				return errors.RequiredNotSet(string(mi.Desc.Fields().ByNumber(f.num).FullName()))
			}
			continue
		}
		if f.funcs.isInit == nil {
			continue
		}
		if err := f.funcs.isInit(fptr, f); err != nil {
			return err
		}
	}
	return nil
}

func (mi *MessageInfo) isInitExtensions(ext *map[int32]ExtensionField) error {
	if ext == nil {
		return nil
	}
	for _, x := range *ext {
		ei := getExtensionFieldInfo(x.Type())
		if ei.funcs.isInit == nil {
			continue
		}
		v := x.Value()
		if !v.IsValid() {
			continue
		}
		if err := ei.funcs.isInit(v); err != nil {
			return err
		}
	}
	return nil
}

var (
	needsInitCheckMu  sync.Mutex
	needsInitCheckMap sync.Map
)

// needsInitCheck reports whether a message needs to be checked for partial initialization.
//
// It returns true if the message transitively includes any required or extension fields.
func needsInitCheck(md protoreflect.MessageDescriptor) bool {
	if v, ok := needsInitCheckMap.Load(md); ok {
		if has, ok := v.(bool); ok {
			return has
		}
	}
	needsInitCheckMu.Lock()
	defer needsInitCheckMu.Unlock()
	return needsInitCheckLocked(md)
}

func needsInitCheckLocked(md protoreflect.MessageDescriptor) (has bool) {
	if v, ok := needsInitCheckMap.Load(md); ok {
		// If has is true, we've previously determined that this message
		// needs init checks.
		//
		// If has is false, we've previously determined that it can never
		// be uninitialized.
		//
		// If has is not a bool, we've just encountered a cycle in the
		// message graph. In this case, it is safe to return false: If
		// the message does have required fields, we'll detect them later
		// in the graph traversal.
		has, ok := v.(bool)
		return ok && has
	}
	needsInitCheckMap.Store(md, struct{}{}) // avoid cycles while descending into this message
	defer func() {
		needsInitCheckMap.Store(md, has)
	}()
	if md.RequiredNumbers().Len() > 0 {
		return true
	}
	if md.ExtensionRanges().Len() > 0 {
		return true
	}
	for i := 0; i < md.Fields().Len(); i++ {
		fd := md.Fields().Get(i)
		// Map keys are never messages, so just consider the map value.
		if fd.IsMap() {
			fd = fd.MapValue()
		}
		fmd := fd.Message()
		if fmd != nil && needsInitCheckLocked(fmd) {
			return true
		}
	}
	return false
}
