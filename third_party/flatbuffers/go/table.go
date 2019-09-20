package flatbuffers

// Table wraps a byte slice and provides read access to its data.
//
// The variable `Pos` indicates the root of the FlatBuffers object therein.
type Table struct {
	Bytes []byte
	Pos   UOffsetT // Always < 1<<31.
}

// Offset provides access into the Table's vtable.
//
// Fields which are deprecated are ignored by checking against the vtable's length.
func (t *Table) Offset(vtableOffset VOffsetT) VOffsetT {
	vtable := UOffsetT(SOffsetT(t.Pos) - t.GetSOffsetT(t.Pos))
	if vtableOffset < t.GetVOffsetT(vtable) {
		return t.GetVOffsetT(vtable + UOffsetT(vtableOffset))
	}
	return 0
}

// Indirect retrieves the relative offset stored at `offset`.
func (t *Table) Indirect(off UOffsetT) UOffsetT {
	return off + GetUOffsetT(t.Bytes[off:])
}

// String gets a string from data stored inside the flatbuffer.
func (t *Table) String(off UOffsetT) string {
	b := t.ByteVector(off)
	return byteSliceToString(b)
}

// ByteVector gets a byte slice from data stored inside the flatbuffer.
func (t *Table) ByteVector(off UOffsetT) []byte {
	off += GetUOffsetT(t.Bytes[off:])
	start := off + UOffsetT(SizeUOffsetT)
	length := GetUOffsetT(t.Bytes[off:])
	return t.Bytes[start : start+length]
}

// VectorLen retrieves the length of the vector whose offset is stored at
// "off" in this object.
func (t *Table) VectorLen(off UOffsetT) int {
	off += t.Pos
	off += GetUOffsetT(t.Bytes[off:])
	return int(GetUOffsetT(t.Bytes[off:]))
}

// Vector retrieves the start of data of the vector whose offset is stored
// at "off" in this object.
func (t *Table) Vector(off UOffsetT) UOffsetT {
	off += t.Pos
	x := off + GetUOffsetT(t.Bytes[off:])
	// data starts after metadata containing the vector length
	x += UOffsetT(SizeUOffsetT)
	return x
}

// Union initializes any Table-derived type to point to the union at the given
// offset.
func (t *Table) Union(t2 *Table, off UOffsetT) {
	off += t.Pos
	t2.Pos = off + t.GetUOffsetT(off)
	t2.Bytes = t.Bytes
}

// GetBool retrieves a bool at the given offset.
func (t *Table) GetBool(off UOffsetT) bool {
	return GetBool(t.Bytes[off:])
}

// GetByte retrieves a byte at the given offset.
func (t *Table) GetByte(off UOffsetT) byte {
	return GetByte(t.Bytes[off:])
}

// GetUint8 retrieves a uint8 at the given offset.
func (t *Table) GetUint8(off UOffsetT) uint8 {
	return GetUint8(t.Bytes[off:])
}

// GetUint16 retrieves a uint16 at the given offset.
func (t *Table) GetUint16(off UOffsetT) uint16 {
	return GetUint16(t.Bytes[off:])
}

// GetUint32 retrieves a uint32 at the given offset.
func (t *Table) GetUint32(off UOffsetT) uint32 {
	return GetUint32(t.Bytes[off:])
}

// GetUint64 retrieves a uint64 at the given offset.
func (t *Table) GetUint64(off UOffsetT) uint64 {
	return GetUint64(t.Bytes[off:])
}

// GetInt8 retrieves a int8 at the given offset.
func (t *Table) GetInt8(off UOffsetT) int8 {
	return GetInt8(t.Bytes[off:])
}

// GetInt16 retrieves a int16 at the given offset.
func (t *Table) GetInt16(off UOffsetT) int16 {
	return GetInt16(t.Bytes[off:])
}

// GetInt32 retrieves a int32 at the given offset.
func (t *Table) GetInt32(off UOffsetT) int32 {
	return GetInt32(t.Bytes[off:])
}

// GetInt64 retrieves a int64 at the given offset.
func (t *Table) GetInt64(off UOffsetT) int64 {
	return GetInt64(t.Bytes[off:])
}

// GetFloat32 retrieves a float32 at the given offset.
func (t *Table) GetFloat32(off UOffsetT) float32 {
	return GetFloat32(t.Bytes[off:])
}

// GetFloat64 retrieves a float64 at the given offset.
func (t *Table) GetFloat64(off UOffsetT) float64 {
	return GetFloat64(t.Bytes[off:])
}

// GetUOffsetT retrieves a UOffsetT at the given offset.
func (t *Table) GetUOffsetT(off UOffsetT) UOffsetT {
	return GetUOffsetT(t.Bytes[off:])
}

// GetVOffsetT retrieves a VOffsetT at the given offset.
func (t *Table) GetVOffsetT(off UOffsetT) VOffsetT {
	return GetVOffsetT(t.Bytes[off:])
}

// GetSOffsetT retrieves a SOffsetT at the given offset.
func (t *Table) GetSOffsetT(off UOffsetT) SOffsetT {
	return GetSOffsetT(t.Bytes[off:])
}

// GetBoolSlot retrieves the bool that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetBoolSlot(slot VOffsetT, d bool) bool {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetBool(t.Pos + UOffsetT(off))
}

// GetByteSlot retrieves the byte that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetByteSlot(slot VOffsetT, d byte) byte {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetByte(t.Pos + UOffsetT(off))
}

// GetInt8Slot retrieves the int8 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetInt8Slot(slot VOffsetT, d int8) int8 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetInt8(t.Pos + UOffsetT(off))
}

// GetUint8Slot retrieves the uint8 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetUint8Slot(slot VOffsetT, d uint8) uint8 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetUint8(t.Pos + UOffsetT(off))
}

// GetInt16Slot retrieves the int16 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetInt16Slot(slot VOffsetT, d int16) int16 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetInt16(t.Pos + UOffsetT(off))
}

// GetUint16Slot retrieves the uint16 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetUint16Slot(slot VOffsetT, d uint16) uint16 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetUint16(t.Pos + UOffsetT(off))
}

// GetInt32Slot retrieves the int32 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetInt32Slot(slot VOffsetT, d int32) int32 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetInt32(t.Pos + UOffsetT(off))
}

// GetUint32Slot retrieves the uint32 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetUint32Slot(slot VOffsetT, d uint32) uint32 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetUint32(t.Pos + UOffsetT(off))
}

// GetInt64Slot retrieves the int64 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetInt64Slot(slot VOffsetT, d int64) int64 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetInt64(t.Pos + UOffsetT(off))
}

// GetUint64Slot retrieves the uint64 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetUint64Slot(slot VOffsetT, d uint64) uint64 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetUint64(t.Pos + UOffsetT(off))
}

// GetFloat32Slot retrieves the float32 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetFloat32Slot(slot VOffsetT, d float32) float32 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetFloat32(t.Pos + UOffsetT(off))
}

// GetFloat64Slot retrieves the float64 that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetFloat64Slot(slot VOffsetT, d float64) float64 {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}

	return t.GetFloat64(t.Pos + UOffsetT(off))
}

// GetVOffsetTSlot retrieves the VOffsetT that the given vtable location
// points to. If the vtable value is zero, the default value `d`
// will be returned.
func (t *Table) GetVOffsetTSlot(slot VOffsetT, d VOffsetT) VOffsetT {
	off := t.Offset(slot)
	if off == 0 {
		return d
	}
	return VOffsetT(off)
}

// MutateBool updates a bool at the given offset.
func (t *Table) MutateBool(off UOffsetT, n bool) bool {
	WriteBool(t.Bytes[off:], n)
	return true
}

// MutateByte updates a Byte at the given offset.
func (t *Table) MutateByte(off UOffsetT, n byte) bool {
	WriteByte(t.Bytes[off:], n)
	return true
}

// MutateUint8 updates a Uint8 at the given offset.
func (t *Table) MutateUint8(off UOffsetT, n uint8) bool {
	WriteUint8(t.Bytes[off:], n)
	return true
}

// MutateUint16 updates a Uint16 at the given offset.
func (t *Table) MutateUint16(off UOffsetT, n uint16) bool {
	WriteUint16(t.Bytes[off:], n)
	return true
}

// MutateUint32 updates a Uint32 at the given offset.
func (t *Table) MutateUint32(off UOffsetT, n uint32) bool {
	WriteUint32(t.Bytes[off:], n)
	return true
}

// MutateUint64 updates a Uint64 at the given offset.
func (t *Table) MutateUint64(off UOffsetT, n uint64) bool {
	WriteUint64(t.Bytes[off:], n)
	return true
}

// MutateInt8 updates a Int8 at the given offset.
func (t *Table) MutateInt8(off UOffsetT, n int8) bool {
	WriteInt8(t.Bytes[off:], n)
	return true
}

// MutateInt16 updates a Int16 at the given offset.
func (t *Table) MutateInt16(off UOffsetT, n int16) bool {
	WriteInt16(t.Bytes[off:], n)
	return true
}

// MutateInt32 updates a Int32 at the given offset.
func (t *Table) MutateInt32(off UOffsetT, n int32) bool {
	WriteInt32(t.Bytes[off:], n)
	return true
}

// MutateInt64 updates a Int64 at the given offset.
func (t *Table) MutateInt64(off UOffsetT, n int64) bool {
	WriteInt64(t.Bytes[off:], n)
	return true
}

// MutateFloat32 updates a Float32 at the given offset.
func (t *Table) MutateFloat32(off UOffsetT, n float32) bool {
	WriteFloat32(t.Bytes[off:], n)
	return true
}

// MutateFloat64 updates a Float64 at the given offset.
func (t *Table) MutateFloat64(off UOffsetT, n float64) bool {
	WriteFloat64(t.Bytes[off:], n)
	return true
}

// MutateUOffsetT updates a UOffsetT at the given offset.
func (t *Table) MutateUOffsetT(off UOffsetT, n UOffsetT) bool {
	WriteUOffsetT(t.Bytes[off:], n)
	return true
}

// MutateVOffsetT updates a VOffsetT at the given offset.
func (t *Table) MutateVOffsetT(off UOffsetT, n VOffsetT) bool {
	WriteVOffsetT(t.Bytes[off:], n)
	return true
}

// MutateSOffsetT updates a SOffsetT at the given offset.
func (t *Table) MutateSOffsetT(off UOffsetT, n SOffsetT) bool {
	WriteSOffsetT(t.Bytes[off:], n)
	return true
}

// MutateBoolSlot updates the bool at given vtable location
func (t *Table) MutateBoolSlot(slot VOffsetT, n bool) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateBool(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateByteSlot updates the byte at given vtable location
func (t *Table) MutateByteSlot(slot VOffsetT, n byte) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateByte(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateInt8Slot updates the int8 at given vtable location
func (t *Table) MutateInt8Slot(slot VOffsetT, n int8) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateInt8(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateUint8Slot updates the uint8 at given vtable location
func (t *Table) MutateUint8Slot(slot VOffsetT, n uint8) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateUint8(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateInt16Slot updates the int16 at given vtable location
func (t *Table) MutateInt16Slot(slot VOffsetT, n int16) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateInt16(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateUint16Slot updates the uint16 at given vtable location
func (t *Table) MutateUint16Slot(slot VOffsetT, n uint16) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateUint16(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateInt32Slot updates the int32 at given vtable location
func (t *Table) MutateInt32Slot(slot VOffsetT, n int32) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateInt32(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateUint32Slot updates the uint32 at given vtable location
func (t *Table) MutateUint32Slot(slot VOffsetT, n uint32) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateUint32(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateInt64Slot updates the int64 at given vtable location
func (t *Table) MutateInt64Slot(slot VOffsetT, n int64) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateInt64(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateUint64Slot updates the uint64 at given vtable location
func (t *Table) MutateUint64Slot(slot VOffsetT, n uint64) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateUint64(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateFloat32Slot updates the float32 at given vtable location
func (t *Table) MutateFloat32Slot(slot VOffsetT, n float32) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateFloat32(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}

// MutateFloat64Slot updates the float64 at given vtable location
func (t *Table) MutateFloat64Slot(slot VOffsetT, n float64) bool {
	if off := t.Offset(slot); off != 0 {
		t.MutateFloat64(t.Pos+UOffsetT(off), n)
		return true
	}

	return false
}
