package tensorflow

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/golang/protobuf/proto"
)

type Session struct {
	session TF_Session
}

// NewSession initializes a new TensorFlow session.
func NewSession() (*Session, error) {
	status := TF_NewStatus()
	return &Session{
		session: TF_NewSession(
			TF_NewSessionOptions(),
			status,
		),
	}, statusToError(status)
}

func encodeStrings(in []string) []byte {
	size := 0
	for _, s := range in {
		size += 8
		size += len(s)
		size += len(proto.EncodeVarint(uint64(len(s))))
	}

	out := make([]byte, size)

	dataPos := 8 * len(in)
	data := out[dataPos:]
	offset := 0
	for i, s := range in {
		inBytes := []byte(s)
		binary.LittleEndian.PutUint64(out[i*8:], uint64(offset))
		inLen := proto.EncodeVarint(uint64(len(s)))
		offset += copy(data[offset:], inLen)
		offset += copy(data[offset:], inBytes)
	}
	return out
}

func Constant(value interface{}) (*Tensor, error) {
	switch v := value.(type) {
	case string:
		buf := encodeStrings([]string{v})
		t, err := newTensor(DataType_DT_STRING, tensorShapeScalar, uintptr(unsafe.Pointer(&(buf[0]))), int64(len(buf)))
		if err != nil {
			return nil, err
		}
		return t, nil
	default:
		return nil, fmt.Errorf("tensorflow: unsupported type %T", value)
	}
}

func statusToError(status TF_Status) error {
	code := TF_GetCode(status)
	message := TF_Message(status)

	if code != 0 {
		return fmt.Errorf("tensorflow: %d: %v", code, message)
	}
	return nil
}

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

	TF_Run_wrapper(s.session, inputNames, inputValues, outputNames, outputValues, targetNames, status)

	result := make([]*Tensor, 0, outputValues.Size())
	for i := int64(0); i < outputValues.Size(); i++ {
		result = append(result, &Tensor{
			tensor: outputValues.Get(int(i)),
		})
	}
	return result, statusToError(status)
}

func (s *Session) ExtendGraph(graph *GraphDef) error {
	status := TF_NewStatus()
	buf, err := proto.Marshal(graph)
	if err != nil {
		return err
	}
	TF_ExtendGraph(s.session, buf, status)
	return statusToError(status)
}
