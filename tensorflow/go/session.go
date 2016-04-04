package tensorflow

import (
	"fmt"

	"github.com/golang/protobuf/proto"
)

// Session A Session instance lets a caller drive a TensorFlow graph
// computation.
type Session struct {
	session TF_Session
}

// NewSession initializes a new TensorFlow session.
func NewSession() (s *Session, err error) {
	status := TF_NewStatus()

	s = &Session{
		session: TF_NewSession(
			TF_NewSessionOptions(),
			status,
		),
	}
	err = s.statusToError(status)

	return
}

// Run Runs the operations on the target nodes, or all the operations if not
// targets are specified. the Parameter Input in a dictionary where the key is
// the tensor name on the graph, and the value the Tensor. The parameter
// outputs is used to specify the tensors from the graph to be returned in the
// same order as they occur on the slice.
func (s *Session) Run(inputs map[string]*Tensor, outputs []string, targets []string) ([]*Tensor, error) {
	inputNames := NewStringVector()
	inputValues := NewTensorVector()
	for k, v := range inputs {
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

// ExtendGraph Loads the graph definition on the session
func (s *Session) ExtendGraph(graph *Graph) error {
	status := TF_NewStatus()
	buf, err := proto.Marshal(graph.def)
	if err != nil {
		return err
	}
	TF_ExtendGraph(s.session, buf, status)

	return s.statusToError(status)
}

// ErrStatusTf Error message comming out from the TensorFlow C++ libraries
type ErrStatusTf struct {
	code    TF_Code
	message string
}

func (e *ErrStatusTf) Error() string {
	return fmt.Sprintf("tensorflow: %d: %v", e.code, e.message)
}

// statusToError Converts a TF_Status returned by a C execution into a Go Error
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
