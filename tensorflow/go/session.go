/*
Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tensorflow

// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
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
	cOpt, doneOpt, err := options.c()
	defer doneOpt()
	if err != nil {
		return nil, err
	}
	cSess := C.TF_NewSession(graph.c, cOpt, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}

	s := &Session{c: cSess}
	runtime.SetFinalizer(s, func(s *Session) { s.Close() })
	return s, nil
}

// Device structure contains information about a device associated with a session, as returned by ListDevices()
type Device struct {
	Name, Type       string
	MemoryLimitBytes int64
}

// String describes d and implements fmt.Stringer.
func (d Device) String() string {
	memStr := "no memory limit"
	if d.MemoryLimitBytes >= 0 {
		memStr = fmt.Sprintf("memory limit %d bytes", d.MemoryLimitBytes)
	}
	return fmt.Sprintf("(Device: name \"%s\", type %s, %s)", d.Name, d.Type, memStr)
}

func deviceSliceFromDeviceList(list *C.TF_DeviceList) ([]Device, error) {
	var devices []Device
	status := newStatus()

	for i := 0; i < int(C.TF_DeviceListCount(list)); i++ {
		name := C.TF_DeviceListName(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListName(index=%d) failed: %v", i, err)
		}

		deviceType := C.TF_DeviceListType(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListType(index=%d) failed: %v", i, err)
		}

		memoryLimitBytes := C.TF_DeviceListMemoryBytes(list, C.int(i), status.c)
		if err := status.Err(); err != nil {
			return nil, fmt.Errorf("DeviceListMemoryBytes(index=%d) failed: %v", i, err)
		}

		device := Device{
			Name:             C.GoString(name),
			Type:             C.GoString(deviceType),
			MemoryLimitBytes: int64(memoryLimitBytes),
		}

		devices = append(devices, device)
	}

	return devices, nil
}

// ListDevices returns the list of devices associated with a Session.
func (s *Session) ListDevices() ([]Device, error) {
	status := newStatus()
	devicesList := C.TF_SessionListDevices(s.c, status.c)
	if err := status.Err(); err != nil {
		return nil, fmt.Errorf("SessionListDevices() failed: %v", err)
	}
	defer C.TF_DeleteDeviceList(devicesList)
	return deviceSliceFromDeviceList(devicesList)
}

// Run the graph with the associated session starting with the supplied feeds
// to compute the value of the requested fetches. Runs, but does not return
// Tensors for operations specified in targets.
//
// On success, returns the fetched Tensors in the same order as supplied in
// the fetches argument. If fetches is set to nil, the returned Tensor fetches
// is empty.
func (s *Session) Run(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) ([]*Tensor, error) {
	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	c := newCRunArgs(feeds, fetches, targets)
	status := newStatus()
	C.TF_SessionRun(s.c, nil,
		ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
		ptrOutput(c.fetches), ptrTensor(c.fetchTensors), C.int(len(fetches)),
		ptrOperation(c.targets), C.int(len(targets)),
		nil, status.c)

	// Make sure GC won't harvest input tensors until SessionRun() is finished
	runtime.KeepAlive(feeds)

	if err := status.Err(); err != nil {
		return nil, err
	}
	return c.toGo(), nil
}

// PartialRun enables incremental evaluation of graphs.
//
// PartialRun allows the caller to pause the evaluation of a graph, run
// arbitrary code that depends on the intermediate computation of the graph,
// and then resume graph execution. The results of the arbitrary code can be
// fed into the graph when resuming execution.  In contrast, Session.Run
// executes the graph to compute the requested fetches using the provided feeds
// and discards all intermediate state (e.g., value of intermediate tensors)
// when it returns.
//
// For example, consider a graph for unsupervised training of a neural network
// model. PartialRun can be used to pause execution after the forward pass of
// the network, let the caller actuate the output (e.g., play a game, actuate a
// robot etc.), determine the error/loss and then feed this calculated loss
// when resuming the backward pass of the graph.
type PartialRun struct {
	session *Session
	handle  *C.char
}

// Run resumes execution of the graph to compute the requested fetches and
// targets with the provided feeds.
func (pr *PartialRun) Run(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) ([]*Tensor, error) {
	var (
		c      = newCRunArgs(feeds, fetches, targets)
		status = newStatus()
		s      = pr.session
	)
	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	C.TF_SessionPRun(s.c, pr.handle,
		ptrOutput(c.feeds), ptrTensor(c.feedTensors), C.int(len(feeds)),
		ptrOutput(c.fetches), ptrTensor(c.fetchTensors), C.int(len(fetches)),
		ptrOperation(c.targets), C.int(len(targets)),
		status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	return c.toGo(), nil
}

// NewPartialRun sets up the graph for incremental evaluation.
//
// All values of feeds, fetches and targets that may be provided to Run calls
// on the returned PartialRun need to be provided to NewPartialRun.
//
// See documentation for the PartialRun type.
func (s *Session) NewPartialRun(feeds, fetches []Output, targets []*Operation) (*PartialRun, error) {
	var (
		cfeeds   = make([]C.TF_Output, len(feeds))
		cfetches = make([]C.TF_Output, len(fetches))
		ctargets = make([]*C.TF_Operation, len(targets))

		pcfeeds   *C.TF_Output
		pcfetches *C.TF_Output
		pctargets **C.TF_Operation

		status = newStatus()
	)
	if len(feeds) > 0 {
		pcfeeds = &cfeeds[0]
		for i, o := range feeds {
			cfeeds[i] = o.c()
		}
	}
	if len(fetches) > 0 {
		pcfetches = &cfetches[0]
		for i, o := range fetches {
			cfetches[i] = o.c()
		}
	}
	if len(targets) > 0 {
		pctargets = &ctargets[0]
		for i, o := range targets {
			ctargets[i] = o.c
		}
	}

	s.mu.Lock()
	if s.c == nil {
		s.mu.Unlock()
		return nil, errors.New("session is closed")
	}
	s.wg.Add(1)
	s.mu.Unlock()
	defer s.wg.Done()

	pr := &PartialRun{session: s}
	C.TF_SessionPRunSetup(s.c,
		pcfeeds, C.int(len(feeds)),
		pcfetches, C.int(len(fetches)),
		pctargets, C.int(len(targets)),
		&pr.handle, status.c)
	if err := status.Err(); err != nil {
		return nil, err
	}
	runtime.SetFinalizer(pr, func(pr *PartialRun) {
		C.TF_DeletePRunHandle(pr.handle)
	})
	return pr, nil
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

	// Config is a binary-serialized representation of the
	// tensorflow.ConfigProto protocol message
	// (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto).
	Config []byte
}

// c converts the SessionOptions to the C API's TF_SessionOptions. Callers must
// deallocate by calling the returned done() closure.
func (o *SessionOptions) c() (ret *C.TF_SessionOptions, done func(), err error) {
	opt := C.TF_NewSessionOptions()
	if o == nil {
		return opt, func() { C.TF_DeleteSessionOptions(opt) }, nil
	}
	t := C.CString(o.Target)
	C.TF_SetTarget(opt, t)
	C.free(unsafe.Pointer(t))

	var cConfig unsafe.Pointer
	if sz := len(o.Config); sz > 0 {
		status := newStatus()
		// Copying into C-memory is the simplest thing to do in terms
		// of memory safety and cgo rules ("C code may not keep a copy
		// of a Go pointer after the call returns" from
		// https://golang.org/cmd/cgo/#hdr-Passing_pointers).
		cConfig = C.CBytes(o.Config)
		C.TF_SetConfig(opt, cConfig, C.size_t(sz), status.c)
		if err := status.Err(); err != nil {
			C.TF_DeleteSessionOptions(opt)
			return nil, func() {}, fmt.Errorf("invalid SessionOptions.Config: %v", err)
		}
	}
	return opt, func() {
		C.TF_DeleteSessionOptions(opt)
		C.free(cConfig)
	}, nil
}

// cRunArgs translates the arguments to Session.Run and PartialRun.Run into
// values suitable for C library calls.
type cRunArgs struct {
	feeds        []C.TF_Output
	feedTensors  []*C.TF_Tensor
	fetches      []C.TF_Output
	fetchTensors []*C.TF_Tensor
	targets      []*C.TF_Operation
}

type feedsort struct {
	feeds       []C.TF_Output
	feedTensors []*C.TF_Tensor
}

func (f *feedsort) Less(i, j int) bool {
	// Ideally we would sort on the output names. But that's not easy for us to
	// do efficiently as loads of Go name strings would be created from the C
	// strings and destroyed. But we can sort on the addresses of the operation
	// names. This won't sort alphabetically, but for a given set of feeds it
	// should give consistent results from one run to the next.
	ni := uintptr(unsafe.Pointer(C.TF_OperationName(f.feeds[i].oper)))
	nj := uintptr(unsafe.Pointer(C.TF_OperationName(f.feeds[j].oper)))
	if ni == nj {
		// if the names are the same the index may differ
		return f.feeds[i].index < f.feeds[j].index
	}
	return ni < nj
}

func (f *feedsort) Swap(i, j int) {
	f.feeds[i], f.feeds[j] = f.feeds[j], f.feeds[i]
	f.feedTensors[i], f.feedTensors[j] = f.feedTensors[j], f.feedTensors[i]
}

func (f *feedsort) Len() int {
	return len(f.feeds)
}

func newCRunArgs(feeds map[Output]*Tensor, fetches []Output, targets []*Operation) *cRunArgs {
	c := &cRunArgs{
		fetches:      make([]C.TF_Output, len(fetches)),
		fetchTensors: make([]*C.TF_Tensor, len(fetches)),
		targets:      make([]*C.TF_Operation, len(targets)),
	}
	// Go map iteration order is random. So our list of input names will be
	// random for each Run. This interacts badly with the TF core code which
	// builds a executor cache key from these names in the order we provide
	// them. We'll eventually enumerate every possible order and store it in the
	// executor cache. With n inputs that's n! entries. That gets very big very
	// quickly.
	for o, t := range feeds {
		c.feeds = append(c.feeds, o.c())
		c.feedTensors = append(c.feedTensors, t.c)
	}
	if len(c.feeds) > 1 {
		fs := feedsort{feeds: c.feeds, feedTensors: c.feedTensors}
		sort.Sort(&fs)
	}

	for i, o := range fetches {
		c.fetches[i] = o.c()
	}
	for i, t := range targets {
		c.targets[i] = t.c
	}
	return c
}

func (c *cRunArgs) toGo() []*Tensor {
	ret := make([]*Tensor, len(c.fetchTensors))
	for i, ct := range c.fetchTensors {
		ret[i] = newTensorFromC(ct)
	}
	return ret
}

func ptrOutput(l []C.TF_Output) *C.TF_Output {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}

func ptrTensor(l []*C.TF_Tensor) **C.TF_Tensor {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}

func ptrOperation(l []*C.TF_Operation) **C.TF_Operation {
	if len(l) == 0 {
		return nil
	}
	return &l[0]
}
