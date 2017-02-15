// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// Session drives a TensorFlow graph computation.
//
// When a Session is created with a given target, a new Session object is bound
// to the universe of resources specified by that target. Those resources are
// available to this session to perform computation described in the GraphDef.
// After creating the session with a graph, the caller uses the Run() API to
// perform the computation and potentially fetch outputs as Tensors.
// A Session allows concurrent calls to Run().
type Session struct {
	c *C.TF_Session

	// For ensuring that:
	// - Close() blocks on all Run() calls to complete.
	// - Close() can be called multiple times.
	wg sync.WaitGroup
	mu sync.Mutex
}

// NewSession creates a new execution session with the associated graph.
// options may be nil to use the default options.
func NewSession(graph *Graph, options *SessionOptions) (*Session, error) {
	status := newStatus()
	cOpt := options.c()
	cSess := C.TF_NewSession(graph.c, cOpt, status.c)
	C.TF_DeleteSessionOptions(cOpt)
	if err := status.Err(); err != nil {
		return nil, err
	}

	s := &Session{c: cSess}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

// Run the graph with the associated session starting with the supplied inputs.
// inputs and outputs may be set to nil. Runs, but does not return Tensors
// for operations specified in targets.
//
// On success, returns the Tensor outputs in the same order as supplied in
// the outputs argument. If outputs is set to nil, the returned Tensor outputs
// is empty.
func (s *Session) Run(inputs map[Output]*Tensor, outputs []Output, targets []*Operation) ([]*Tensor, error) {
	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	var inputPorts []C.TF_Output
	var inputValues []*C.TF_Tensor
	if inputs != nil {
		for port, tensor := range inputs {
			inputPorts = append(inputPorts, port.c())
			inputValues = append(inputValues, tensor.c)
		}
	}

	var outputPorts []C.TF_Output
	for _, port := range outputs {
		outputPorts = append(outputPorts, port.c())
	}
	outputValues := make([]*C.TF_Tensor, len(outputs))
	var cTargets []*C.TF_Operation
	for _, target := range targets {
		cTargets = append(cTargets, target.c)
	}

	status := newStatus()
	var inputPortsPtr *C.TF_Output
	var inputValuesPtr **C.TF_Tensor
	if len(inputPorts) > 0 {
		inputPortsPtr = &inputPorts[0]
		inputValuesPtr = &inputValues[0]
	}

	var outputPortsPtr *C.TF_Output
	var outputValuesPtr **C.TF_Tensor
	if len(outputPorts) > 0 {
		outputPortsPtr = &outputPorts[0]
		outputValuesPtr = &outputValues[0]
	}

	var cTargetsPtr **C.TF_Operation
	if len(cTargets) > 0 {
		cTargetsPtr = &cTargets[0]
	}

	C.TF_SessionRun(s.c, nil, inputPortsPtr, inputValuesPtr, C.int(len(inputPorts)), outputPortsPtr, outputValuesPtr, C.int(len(outputPorts)), cTargetsPtr, C.int(len(cTargets)), nil, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	tensors := make([]*Tensor, len(outputValues))
	for i, val := range outputValues {
		tensors[i] = newTensorFromC(val)
	}

	return tensors, nil
}

// Close a session. This contacts any other processes associated with this
// session, if applicable. Blocks until all previous calls to Run have returned.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.wg.Wait()
	if s.c == nil {
		return nil
	}
	status := newStatus()
	C.TF_CloseSession(s.c, status.c)
	if err := status.Err(); err != nil {
		return err
	}
	C.TF_DeleteSession(s.c, status.c)
	s.c = nil
	return status.Err()
}

// SessionOptions contains configuration information for a session.
type SessionOptions struct {
	// Target indicates the TensorFlow runtime to connect to.
	//
	// If 'target' is empty or unspecified, the local TensorFlow runtime
	// implementation will be used.  Otherwise, the TensorFlow engine
	// defined by 'target' will be used to perform all computations.
	//
	// "target" can be either a single entry or a comma separated list
	// of entries. Each entry is a resolvable address of one of the
	// following formats:
	//   local
	//   ip:port
	//   host:port
	//   ... other system-specific formats to identify tasks and jobs ...
	//
	// NOTE: at the moment 'local' maps to an in-process service-based
	// runtime.
	//
	// Upon creation, a single session affines itself to one of the
	// remote processes, with possible load balancing choices when the
	// "target" resolves to a list of possible processes.
	//
	// If the session disconnects from the remote process during its
	// lifetime, session calls may fail immediately.
	Target string
}

// c converts the SessionOptions to the C API's TF_SessionOptions. Callers must
// deallocate by calling C.TF_DeleteSessionOptions().
func (o *SessionOptions) c() *C.TF_SessionOptions {
	opt := C.TF_NewSessionOptions()
	if o == nil {
		return opt
	}
	t := C.CString(o.Target)
	C.TF_SetTarget(opt, t)
	C.free(unsafe.Pointer(t))
	return opt
}
