package flatbuffers

// Codec implements gRPC-go Codec which is used to encode and decode messages.
var Codec = "flatbuffers"

// FlatbuffersCodec defines the interface gRPC uses to encode and decode messages.  Note
// that implementations of this interface must be thread safe; a Codec's
// methods can be called from concurrent goroutines.
type FlatbuffersCodec struct{}

// Marshal returns the wire format of v.
func (FlatbuffersCodec) Marshal(v interface{}) ([]byte, error) {
	return v.(*Builder).FinishedBytes(), nil
}

// Unmarshal parses the wire format into v.
func (FlatbuffersCodec) Unmarshal(data []byte, v interface{}) error {
	v.(flatbuffersInit).Init(data, GetUOffsetT(data))
	return nil
}

// String  old gRPC Codec interface func
func (FlatbuffersCodec) String() string {
	return Codec
}

// Name returns the name of the Codec implementation. The returned string
// will be used as part of content type in transmission.  The result must be
// static; the result cannot change between calls.
//
// add Name() for ForceCodec interface
func (FlatbuffersCodec) Name() string {
	return Codec
}

type flatbuffersInit interface {
	Init(data []byte, i UOffsetT)
}
