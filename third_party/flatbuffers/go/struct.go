package flatbuffers

// Struct wraps a byte slice and provides read access to its data.
//
// Structs do not have a vtable.
type Struct struct {
	Table
}
