package tensorflow

import (
	"fmt"
	"runtime"

	"github.com/golang/protobuf/proto"
)

// A Session instance lets a caller drive a TensorFlow graph computation.
type Session struct {
	ops     TF_SessionOptions
	session TF_Session
	status  TF_Status
	graph   *Graph
}

// NewSession initializes a new TensorFlow session.
func NewSession() (s *Session, err error) {
	status := TF_NewStatus()
	ops := TF_NewSessionOptions()
	s = &Session{
		ops:    ops,
		status: status,
		session: TF_NewSession(
			ops,
			status,
		),
	}

	if err = s.statusToError(status); err != nil {
		return nil, err
	}

	// Release the C allocated memory when the instance is destroyed
	runtime.SetFinalizer(s, (*Session).FreeAllocMem)

	return s, nil
}

// Run runs the operations on the target nodes, or all the operations if not
// targets are specified. the Parameter Input in a dictionary where the key is
// the tensor name on the graph, and the value the Tensor. The parameter
// outputs is used to specify the tensors from the graph to be returned in the
// same order as they occur on the slice.
func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error) {
	inputNames := NewStringVector()
	inputValues := NewTensorVector()
	for k, v := range inputs {
		v.setCMemAsAlreadyRelease()
		inputValues.Add(v.tensor)
		inputNames.Add(k)
	}
	outputNames := NewStringVector()
	for _, n := range outputs {
		outputNames.Add(n)
	}

	targetNames := NewStringVector()
	for _, n := range targets {
		targetNames.Add(n)
	}

	outputValues := NewTensorVector()
	status := TF_NewStatus()
	defer TF_DeleteStatus(status)

	TF_Run_wrapper(
		s.session,
		inputNames,
		inputValues,
		outputNames,
		outputValues,
		targetNames,
		status)

	result := make([]*Tensor, 0, outputValues.Size())
	for i := int64(0); i < outputValues.Size(); i++ {
		result = append(result, &Tensor{
			tensor: outputValues.Get(int(i)),
		})
	}

	return result, s.statusToError(status)
}

// ExtendGraph loads the graph definition on the session.
func (s *Session) ExtendGraph(graph *Graph) (err error) {
	status := TF_NewStatus()
	defer TF_DeleteStatus(status)
	buf, err := proto.Marshal(graph.def)
	if err != nil {
		return err
	}

	TF_ExtendGraph(s.session, buf, status)
	s.graph = graph

	return s.statusToError(status)
}

// ExtendAndInitializeAllVariables adds the "init" op to the graph in order to
// initialize all the variables, loads the graph definition on the session
// and executes the "init" op.
func (s *Session) ExtendAndInitializeAllVariables(graph *Graph) (err error) {
	// Extend the initialization graph, and execute the init op, this will
	// initialize all the variables
	graph.addInitializationGraphOp()
	if err = s.ExtendGraph(graph); err != nil {
		return err
	}
	_, err = s.Run(nil, nil, []string{"init"})

	return nil
}

// FreeAllocMem method defined to be invoked by the Go garbage collector before
// release this instance releasing the C++ allocated memory.
func (s *Session) FreeAllocMem() {
	TF_DeleteSession(s.session, s.status)
	TF_DeleteStatus(s.status)
	TF_DeleteSessionOptions(s.ops)
}

// ErrStatusTf error message comming out from the TensorFlow C++ libraries.
type ErrStatusTf struct {
	code    TF_Code
	message string
}

func (e *ErrStatusTf) Error() string {
	return fmt.Sprintf("tensorflow: %d: %v", e.code, e.message)
}

// statusToError converts a TF_Status returned by a C execution into a Go Error.
func (s *Session) statusToError(status TF_Status) error {
	code := TF_GetCode(status)
	message := TF_Message(status)

	if code != 0 {
		return &ErrStatusTf{
			code:    code,
			message: message,
		}
	}

	return nil
}
