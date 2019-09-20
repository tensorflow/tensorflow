package flatbuffers

// Builder is a state machine for creating FlatBuffer objects.
// Use a Builder to construct object(s) starting from leaf nodes.
//
// A Builder constructs byte buffers in a last-first manner for simplicity and
// performance.
type Builder struct {
	// `Bytes` gives raw access to the buffer. Most users will want to use
	// FinishedBytes() instead.
	Bytes []byte

	minalign  int
	vtable    []UOffsetT
	objectEnd UOffsetT
	vtables   []UOffsetT
	head      UOffsetT
	nested    bool
	finished  bool
}

const fileIdentifierLength = 4

// NewBuilder initializes a Builder of size `initial_size`.
// The internal buffer is grown as needed.
func NewBuilder(initialSize int) *Builder {
	if initialSize <= 0 {
		initialSize = 0
	}

	b := &Builder{}
	b.Bytes = make([]byte, initialSize)
	b.head = UOffsetT(initialSize)
	b.minalign = 1
	b.vtables = make([]UOffsetT, 0, 16) // sensible default capacity

	return b
}

// Reset truncates the underlying Builder buffer, facilitating alloc-free
// reuse of a Builder. It also resets bookkeeping data.
func (b *Builder) Reset() {
	if b.Bytes != nil {
		b.Bytes = b.Bytes[:cap(b.Bytes)]
	}

	if b.vtables != nil {
		b.vtables = b.vtables[:0]
	}

	if b.vtable != nil {
		b.vtable = b.vtable[:0]
	}

	b.head = UOffsetT(len(b.Bytes))
	b.minalign = 1
	b.nested = false
	b.finished = false
}

// FinishedBytes returns a pointer to the written data in the byte buffer.
// Panics if the builder is not in a finished state (which is caused by calling
// `Finish()`).
func (b *Builder) FinishedBytes() []byte {
	b.assertFinished()
	return b.Bytes[b.Head():]
}

// StartObject initializes bookkeeping for writing a new object.
func (b *Builder) StartObject(numfields int) {
	b.assertNotNested()
	b.nested = true

	// use 32-bit offsets so that arithmetic doesn't overflow.
	if cap(b.vtable) < numfields || b.vtable == nil {
		b.vtable = make([]UOffsetT, numfields)
	} else {
		b.vtable = b.vtable[:numfields]
		for i := 0; i < len(b.vtable); i++ {
			b.vtable[i] = 0
		}
	}

	b.objectEnd = b.Offset()
}

// WriteVtable serializes the vtable for the current object, if applicable.
//
// Before writing out the vtable, this checks pre-existing vtables for equality
// to this one. If an equal vtable is found, point the object to the existing
// vtable and return.
//
// Because vtable values are sensitive to alignment of object data, not all
// logically-equal vtables will be deduplicated.
//
// A vtable has the following format:
//   <VOffsetT: size of the vtable in bytes, including this value>
//   <VOffsetT: size of the object in bytes, including the vtable offset>
//   <VOffsetT: offset for a field> * N, where N is the number of fields in
//	        the schema for this type. Includes deprecated fields.
// Thus, a vtable is made of 2 + N elements, each SizeVOffsetT bytes wide.
//
// An object has the following format:
//   <SOffsetT: offset to this object's vtable (may be negative)>
//   <byte: data>+
func (b *Builder) WriteVtable() (n UOffsetT) {
	// Prepend a zero scalar to the object. Later in this function we'll
	// write an offset here that points to the object's vtable:
	b.PrependSOffsetT(0)

	objectOffset := b.Offset()
	existingVtable := UOffsetT(0)

	// Trim vtable of trailing zeroes.
	i := len(b.vtable) - 1
	for ; i >= 0 && b.vtable[i] == 0; i-- {
	}
	b.vtable = b.vtable[:i+1]

	// Search backwards through existing vtables, because similar vtables
	// are likely to have been recently appended. See
	// BenchmarkVtableDeduplication for a case in which this heuristic
	// saves about 30% of the time used in writing objects with duplicate
	// tables.
	for i := len(b.vtables) - 1; i >= 0; i-- {
		// Find the other vtable, which is associated with `i`:
		vt2Offset := b.vtables[i]
		vt2Start := len(b.Bytes) - int(vt2Offset)
		vt2Len := GetVOffsetT(b.Bytes[vt2Start:])

		metadata := VtableMetadataFields * SizeVOffsetT
		vt2End := vt2Start + int(vt2Len)
		vt2 := b.Bytes[vt2Start+metadata : vt2End]

		// Compare the other vtable to the one under consideration.
		// If they are equal, store the offset and break:
		if vtableEqual(b.vtable, objectOffset, vt2) {
			existingVtable = vt2Offset
			break
		}
	}

	if existingVtable == 0 {
		// Did not find a vtable, so write this one to the buffer.

		// Write out the current vtable in reverse , because
		// serialization occurs in last-first order:
		for i := len(b.vtable) - 1; i >= 0; i-- {
			var off UOffsetT
			if b.vtable[i] != 0 {
				// Forward reference to field;
				// use 32bit number to assert no overflow:
				off = objectOffset - b.vtable[i]
			}

			b.PrependVOffsetT(VOffsetT(off))
		}

		// The two metadata fields are written last.

		// First, store the object bytesize:
		objectSize := objectOffset - b.objectEnd
		b.PrependVOffsetT(VOffsetT(objectSize))

		// Second, store the vtable bytesize:
		vBytes := (len(b.vtable) + VtableMetadataFields) * SizeVOffsetT
		b.PrependVOffsetT(VOffsetT(vBytes))

		// Next, write the offset to the new vtable in the
		// already-allocated SOffsetT at the beginning of this object:
		objectStart := SOffsetT(len(b.Bytes)) - SOffsetT(objectOffset)
		WriteSOffsetT(b.Bytes[objectStart:],
			SOffsetT(b.Offset())-SOffsetT(objectOffset))

		// Finally, store this vtable in memory for future
		// deduplication:
		b.vtables = append(b.vtables, b.Offset())
	} else {
		// Found a duplicate vtable.

		objectStart := SOffsetT(len(b.Bytes)) - SOffsetT(objectOffset)
		b.head = UOffsetT(objectStart)

		// Write the offset to the found vtable in the
		// already-allocated SOffsetT at the beginning of this object:
		WriteSOffsetT(b.Bytes[b.head:],
			SOffsetT(existingVtable)-SOffsetT(objectOffset))
	}

	b.vtable = b.vtable[:0]
	return objectOffset
}

// EndObject writes data necessary to finish object construction.
func (b *Builder) EndObject() UOffsetT {
	b.assertNested()
	n := b.WriteVtable()
	b.nested = false
	return n
}

// Doubles the size of the byteslice, and copies the old data towards the
// end of the new byteslice (since we build the buffer backwards).
func (b *Builder) growByteBuffer() {
	if (int64(len(b.Bytes)) & int64(0xC0000000)) != 0 {
		panic("cannot grow buffer beyond 2 gigabytes")
	}
	newLen := len(b.Bytes) * 2
	if newLen == 0 {
		newLen = 1
	}

	if cap(b.Bytes) >= newLen {
		b.Bytes = b.Bytes[:newLen]
	} else {
		extension := make([]byte, newLen-len(b.Bytes))
		b.Bytes = append(b.Bytes, extension...)
	}

	middle := newLen / 2
	copy(b.Bytes[middle:], b.Bytes[:middle])
}

// Head gives the start of useful data in the underlying byte buffer.
// Note: unlike other functions, this value is interpreted as from the left.
func (b *Builder) Head() UOffsetT {
	return b.head
}

// Offset relative to the end of the buffer.
func (b *Builder) Offset() UOffsetT {
	return UOffsetT(len(b.Bytes)) - b.head
}

// Pad places zeros at the current offset.
func (b *Builder) Pad(n int) {
	for i := 0; i < n; i++ {
		b.PlaceByte(0)
	}
}

// Prep prepares to write an element of `size` after `additional_bytes`
// have been written, e.g. if you write a string, you need to align such
// the int length field is aligned to SizeInt32, and the string data follows it
// directly.
// If all you need to do is align, `additionalBytes` will be 0.
func (b *Builder) Prep(size, additionalBytes int) {
	// Track the biggest thing we've ever aligned to.
	if size > b.minalign {
		b.minalign = size
	}
	// Find the amount of alignment needed such that `size` is properly
	// aligned after `additionalBytes`:
	alignSize := (^(len(b.Bytes) - int(b.Head()) + additionalBytes)) + 1
	alignSize &= (size - 1)

	// Reallocate the buffer if needed:
	for int(b.head) <= alignSize+size+additionalBytes {
		oldBufSize := len(b.Bytes)
		b.growByteBuffer()
		b.head += UOffsetT(len(b.Bytes) - oldBufSize)
	}
	b.Pad(alignSize)
}

// PrependSOffsetT prepends an SOffsetT, relative to where it will be written.
func (b *Builder) PrependSOffsetT(off SOffsetT) {
	b.Prep(SizeSOffsetT, 0) // Ensure alignment is already done.
	if !(UOffsetT(off) <= b.Offset()) {
		panic("unreachable: off <= b.Offset()")
	}
	off2 := SOffsetT(b.Offset()) - off + SOffsetT(SizeSOffsetT)
	b.PlaceSOffsetT(off2)
}

// PrependUOffsetT prepends an UOffsetT, relative to where it will be written.
func (b *Builder) PrependUOffsetT(off UOffsetT) {
	b.Prep(SizeUOffsetT, 0) // Ensure alignment is already done.
	if !(off <= b.Offset()) {
		panic("unreachable: off <= b.Offset()")
	}
	off2 := b.Offset() - off + UOffsetT(SizeUOffsetT)
	b.PlaceUOffsetT(off2)
}

// StartVector initializes bookkeeping for writing a new vector.
//
// A vector has the following format:
//   <UOffsetT: number of elements in this vector>
//   <T: data>+, where T is the type of elements of this vector.
func (b *Builder) StartVector(elemSize, numElems, alignment int) UOffsetT {
	b.assertNotNested()
	b.nested = true
	b.Prep(SizeUint32, elemSize*numElems)
	b.Prep(alignment, elemSize*numElems) // Just in case alignment > int.
	return b.Offset()
}

// EndVector writes data necessary to finish vector construction.
func (b *Builder) EndVector(vectorNumElems int) UOffsetT {
	b.assertNested()

	// we already made space for this, so write without PrependUint32
	b.PlaceUOffsetT(UOffsetT(vectorNumElems))

	b.nested = false
	return b.Offset()
}

// CreateString writes a null-terminated string as a vector.
func (b *Builder) CreateString(s string) UOffsetT {
	b.assertNotNested()
	b.nested = true

	b.Prep(int(SizeUOffsetT), (len(s)+1)*SizeByte)
	b.PlaceByte(0)

	l := UOffsetT(len(s))

	b.head -= l
	copy(b.Bytes[b.head:b.head+l], s)

	return b.EndVector(len(s))
}

// CreateByteString writes a byte slice as a string (null-terminated).
func (b *Builder) CreateByteString(s []byte) UOffsetT {
	b.assertNotNested()
	b.nested = true

	b.Prep(int(SizeUOffsetT), (len(s)+1)*SizeByte)
	b.PlaceByte(0)

	l := UOffsetT(len(s))

	b.head -= l
	copy(b.Bytes[b.head:b.head+l], s)

	return b.EndVector(len(s))
}

// CreateByteVector writes a ubyte vector
func (b *Builder) CreateByteVector(v []byte) UOffsetT {
	b.assertNotNested()
	b.nested = true

	b.Prep(int(SizeUOffsetT), len(v)*SizeByte)

	l := UOffsetT(len(v))

	b.head -= l
	copy(b.Bytes[b.head:b.head+l], v)

	return b.EndVector(len(v))
}

func (b *Builder) assertNested() {
	// If you get this assert, you're in an object while trying to write
	// data that belongs outside of an object.
	// To fix this, write non-inline data (like vectors) before creating
	// objects.
	if !b.nested {
		panic("Incorrect creation order: must be inside object.")
	}
}

func (b *Builder) assertNotNested() {
	// If you hit this, you're trying to construct a Table/Vector/String
	// during the construction of its parent table (between the MyTableBuilder
	// and builder.Finish()).
	// Move the creation of these sub-objects to above the MyTableBuilder to
	// not get this assert.
	// Ignoring this assert may appear to work in simple cases, but the reason
	// it is here is that storing objects in-line may cause vtable offsets
	// to not fit anymore. It also leads to vtable duplication.
	if b.nested {
		panic("Incorrect creation order: object must not be nested.")
	}
}

func (b *Builder) assertFinished() {
	// If you get this assert, you're attempting to get access a buffer
	// which hasn't been finished yet. Be sure to call builder.Finish()
	// with your root table.
	// If you really need to access an unfinished buffer, use the Bytes
	// buffer directly.
	if !b.finished {
		panic("Incorrect use of FinishedBytes(): must call 'Finish' first.")
	}
}

// PrependBoolSlot prepends a bool onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependBoolSlot(o int, x, d bool) {
	val := byte(0)
	if x {
		val = 1
	}
	def := byte(0)
	if d {
		def = 1
	}
	b.PrependByteSlot(o, val, def)
}

// PrependByteSlot prepends a byte onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependByteSlot(o int, x, d byte) {
	if x != d {
		b.PrependByte(x)
		b.Slot(o)
	}
}

// PrependUint8Slot prepends a uint8 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependUint8Slot(o int, x, d uint8) {
	if x != d {
		b.PrependUint8(x)
		b.Slot(o)
	}
}

// PrependUint16Slot prepends a uint16 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependUint16Slot(o int, x, d uint16) {
	if x != d {
		b.PrependUint16(x)
		b.Slot(o)
	}
}

// PrependUint32Slot prepends a uint32 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependUint32Slot(o int, x, d uint32) {
	if x != d {
		b.PrependUint32(x)
		b.Slot(o)
	}
}

// PrependUint64Slot prepends a uint64 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependUint64Slot(o int, x, d uint64) {
	if x != d {
		b.PrependUint64(x)
		b.Slot(o)
	}
}

// PrependInt8Slot prepends a int8 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependInt8Slot(o int, x, d int8) {
	if x != d {
		b.PrependInt8(x)
		b.Slot(o)
	}
}

// PrependInt16Slot prepends a int16 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependInt16Slot(o int, x, d int16) {
	if x != d {
		b.PrependInt16(x)
		b.Slot(o)
	}
}

// PrependInt32Slot prepends a int32 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependInt32Slot(o int, x, d int32) {
	if x != d {
		b.PrependInt32(x)
		b.Slot(o)
	}
}

// PrependInt64Slot prepends a int64 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependInt64Slot(o int, x, d int64) {
	if x != d {
		b.PrependInt64(x)
		b.Slot(o)
	}
}

// PrependFloat32Slot prepends a float32 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependFloat32Slot(o int, x, d float32) {
	if x != d {
		b.PrependFloat32(x)
		b.Slot(o)
	}
}

// PrependFloat64Slot prepends a float64 onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependFloat64Slot(o int, x, d float64) {
	if x != d {
		b.PrependFloat64(x)
		b.Slot(o)
	}
}

// PrependUOffsetTSlot prepends an UOffsetT onto the object at vtable slot `o`.
// If value `x` equals default `d`, then the slot will be set to zero and no
// other data will be written.
func (b *Builder) PrependUOffsetTSlot(o int, x, d UOffsetT) {
	if x != d {
		b.PrependUOffsetT(x)
		b.Slot(o)
	}
}

// PrependStructSlot prepends a struct onto the object at vtable slot `o`.
// Structs are stored inline, so nothing additional is being added.
// In generated code, `d` is always 0.
func (b *Builder) PrependStructSlot(voffset int, x, d UOffsetT) {
	if x != d {
		b.assertNested()
		if x != b.Offset() {
			panic("inline data write outside of object")
		}
		b.Slot(voffset)
	}
}

// Slot sets the vtable key `voffset` to the current location in the buffer.
func (b *Builder) Slot(slotnum int) {
	b.vtable[slotnum] = UOffsetT(b.Offset())
}

// FinishWithFileIdentifier finalizes a buffer, pointing to the given `rootTable`.
// as well as applys a file identifier
func (b *Builder) FinishWithFileIdentifier(rootTable UOffsetT, fid []byte) {
	if fid == nil || len(fid) != fileIdentifierLength {
		panic("incorrect file identifier length")
	}
	// In order to add a file identifier to the flatbuffer message, we need
	// to prepare an alignment and file identifier length
	b.Prep(b.minalign, SizeInt32+fileIdentifierLength)
	for i := fileIdentifierLength - 1; i >= 0; i-- {
		// place the file identifier
		b.PlaceByte(fid[i])
	}
	// finish
	b.Finish(rootTable)
}

// Finish finalizes a buffer, pointing to the given `rootTable`.
func (b *Builder) Finish(rootTable UOffsetT) {
	b.assertNotNested()
	b.Prep(b.minalign, SizeUOffsetT)
	b.PrependUOffsetT(rootTable)
	b.finished = true
}

// vtableEqual compares an unwritten vtable to a written vtable.
func vtableEqual(a []UOffsetT, objectStart UOffsetT, b []byte) bool {
	if len(a)*SizeVOffsetT != len(b) {
		return false
	}

	for i := 0; i < len(a); i++ {
		x := GetVOffsetT(b[i*SizeVOffsetT : (i+1)*SizeVOffsetT])

		// Skip vtable entries that indicate a default value.
		if x == 0 && a[i] == 0 {
			continue
		}

		y := SOffsetT(objectStart) - SOffsetT(a[i])
		if SOffsetT(x) != y {
			return false
		}
	}
	return true
}

// PrependBool prepends a bool to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependBool(x bool) {
	b.Prep(SizeBool, 0)
	b.PlaceBool(x)
}

// PrependUint8 prepends a uint8 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependUint8(x uint8) {
	b.Prep(SizeUint8, 0)
	b.PlaceUint8(x)
}

// PrependUint16 prepends a uint16 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependUint16(x uint16) {
	b.Prep(SizeUint16, 0)
	b.PlaceUint16(x)
}

// PrependUint32 prepends a uint32 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependUint32(x uint32) {
	b.Prep(SizeUint32, 0)
	b.PlaceUint32(x)
}

// PrependUint64 prepends a uint64 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependUint64(x uint64) {
	b.Prep(SizeUint64, 0)
	b.PlaceUint64(x)
}

// PrependInt8 prepends a int8 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependInt8(x int8) {
	b.Prep(SizeInt8, 0)
	b.PlaceInt8(x)
}

// PrependInt16 prepends a int16 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependInt16(x int16) {
	b.Prep(SizeInt16, 0)
	b.PlaceInt16(x)
}

// PrependInt32 prepends a int32 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependInt32(x int32) {
	b.Prep(SizeInt32, 0)
	b.PlaceInt32(x)
}

// PrependInt64 prepends a int64 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependInt64(x int64) {
	b.Prep(SizeInt64, 0)
	b.PlaceInt64(x)
}

// PrependFloat32 prepends a float32 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependFloat32(x float32) {
	b.Prep(SizeFloat32, 0)
	b.PlaceFloat32(x)
}

// PrependFloat64 prepends a float64 to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependFloat64(x float64) {
	b.Prep(SizeFloat64, 0)
	b.PlaceFloat64(x)
}

// PrependByte prepends a byte to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependByte(x byte) {
	b.Prep(SizeByte, 0)
	b.PlaceByte(x)
}

// PrependVOffsetT prepends a VOffsetT to the Builder buffer.
// Aligns and checks for space.
func (b *Builder) PrependVOffsetT(x VOffsetT) {
	b.Prep(SizeVOffsetT, 0)
	b.PlaceVOffsetT(x)
}

// PlaceBool prepends a bool to the Builder, without checking for space.
func (b *Builder) PlaceBool(x bool) {
	b.head -= UOffsetT(SizeBool)
	WriteBool(b.Bytes[b.head:], x)
}

// PlaceUint8 prepends a uint8 to the Builder, without checking for space.
func (b *Builder) PlaceUint8(x uint8) {
	b.head -= UOffsetT(SizeUint8)
	WriteUint8(b.Bytes[b.head:], x)
}

// PlaceUint16 prepends a uint16 to the Builder, without checking for space.
func (b *Builder) PlaceUint16(x uint16) {
	b.head -= UOffsetT(SizeUint16)
	WriteUint16(b.Bytes[b.head:], x)
}

// PlaceUint32 prepends a uint32 to the Builder, without checking for space.
func (b *Builder) PlaceUint32(x uint32) {
	b.head -= UOffsetT(SizeUint32)
	WriteUint32(b.Bytes[b.head:], x)
}

// PlaceUint64 prepends a uint64 to the Builder, without checking for space.
func (b *Builder) PlaceUint64(x uint64) {
	b.head -= UOffsetT(SizeUint64)
	WriteUint64(b.Bytes[b.head:], x)
}

// PlaceInt8 prepends a int8 to the Builder, without checking for space.
func (b *Builder) PlaceInt8(x int8) {
	b.head -= UOffsetT(SizeInt8)
	WriteInt8(b.Bytes[b.head:], x)
}

// PlaceInt16 prepends a int16 to the Builder, without checking for space.
func (b *Builder) PlaceInt16(x int16) {
	b.head -= UOffsetT(SizeInt16)
	WriteInt16(b.Bytes[b.head:], x)
}

// PlaceInt32 prepends a int32 to the Builder, without checking for space.
func (b *Builder) PlaceInt32(x int32) {
	b.head -= UOffsetT(SizeInt32)
	WriteInt32(b.Bytes[b.head:], x)
}

// PlaceInt64 prepends a int64 to the Builder, without checking for space.
func (b *Builder) PlaceInt64(x int64) {
	b.head -= UOffsetT(SizeInt64)
	WriteInt64(b.Bytes[b.head:], x)
}

// PlaceFloat32 prepends a float32 to the Builder, without checking for space.
func (b *Builder) PlaceFloat32(x float32) {
	b.head -= UOffsetT(SizeFloat32)
	WriteFloat32(b.Bytes[b.head:], x)
}

// PlaceFloat64 prepends a float64 to the Builder, without checking for space.
func (b *Builder) PlaceFloat64(x float64) {
	b.head -= UOffsetT(SizeFloat64)
	WriteFloat64(b.Bytes[b.head:], x)
}

// PlaceByte prepends a byte to the Builder, without checking for space.
func (b *Builder) PlaceByte(x byte) {
	b.head -= UOffsetT(SizeByte)
	WriteByte(b.Bytes[b.head:], x)
}

// PlaceVOffsetT prepends a VOffsetT to the Builder, without checking for space.
func (b *Builder) PlaceVOffsetT(x VOffsetT) {
	b.head -= UOffsetT(SizeVOffsetT)
	WriteVOffsetT(b.Bytes[b.head:], x)
}

// PlaceSOffsetT prepends a SOffsetT to the Builder, without checking for space.
func (b *Builder) PlaceSOffsetT(x SOffsetT) {
	b.head -= UOffsetT(SizeSOffsetT)
	WriteSOffsetT(b.Bytes[b.head:], x)
}

// PlaceUOffsetT prepends a UOffsetT to the Builder, without checking for space.
func (b *Builder) PlaceUOffsetT(x UOffsetT) {
	b.head -= UOffsetT(SizeUOffsetT)
	WriteUOffsetT(b.Bytes[b.head:], x)
}
