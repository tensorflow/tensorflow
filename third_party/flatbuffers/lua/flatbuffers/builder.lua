local N = require("flatbuffers.numTypes")
local ba = require("flatbuffers.binaryarray")
local compat = require("flatbuffers.compat")

local m = {}

local mt = {}

-- get locals for faster access
local VOffsetT  = N.VOffsetT
local UOffsetT  = N.UOffsetT
local SOffsetT  = N.SOffsetT
local Bool      = N.Bool
local Uint8     = N.Uint8
local Uint16    = N.Uint16
local Uint32    = N.Uint32
local Uint64    = N.Uint64
local Int8      = N.Int8
local Int16     = N.Int16
local Int32     = N.Int32
local Int64     = N.Int64
local Float32   = N.Float32
local Float64   = N.Float64

local MAX_BUFFER_SIZE = 0x80000000 -- 2 GB
local VtableMetadataFields = 2

local getAlignSize = compat.GetAlignSize

local function vtableEqual(a, objectStart, b)
    UOffsetT:EnforceNumber(objectStart)
    if (#a * VOffsetT.bytewidth) ~= #b then
        return false
    end

    for i, elem in ipairs(a) do
        local x = string.unpack(VOffsetT.packFmt, b, 1 + (i - 1) * VOffsetT.bytewidth)
        if x ~= 0 or elem ~= 0 then
            local y = objectStart - elem
            if x ~= y then
                return false
            end
        end
    end
    return true
end

function m.New(initialSize)
    assert(0 <= initialSize and initialSize < MAX_BUFFER_SIZE)
    local o =
    {
        finished = false,
        bytes = ba.New(initialSize),
        nested = false,
        head = initialSize,
        minalign = 1,
        vtables = {}
    }
    setmetatable(o, {__index = mt})
    return o
end

function mt:Output(full)
    assert(self.finished, "Builder Not Finished")
    if full then
        return self.bytes:Slice()
    else
        return self.bytes:Slice(self.head)
    end
end

function mt:StartObject(numFields)
    assert(not self.nested)

    local vtable = {}

    for _=1,numFields do
        table.insert(vtable, 0)
    end

    self.currentVTable = vtable
    self.objectEnd = self:Offset()
    self.nested = true
end

function mt:WriteVtable()
    self:PrependSOffsetTRelative(0)
    local objectOffset = self:Offset()

    local exisitingVTable
    local i = #self.vtables
    while i >= 1 do
        if self.vtables[i] == 0 then
            table.remove(self.vtables,i)
        end
        i = i - 1
    end

    i = #self.vtables
    while i >= 1 do

        local vt2Offset = self.vtables[i]
        local vt2Start = #self.bytes - vt2Offset
        local vt2lenstr = self.bytes:Slice(vt2Start, vt2Start+1)
        local vt2Len = string.unpack(VOffsetT.packFmt, vt2lenstr, 1)

        local metadata = VtableMetadataFields * VOffsetT.bytewidth
        local vt2End = vt2Start + vt2Len
        local vt2 = self.bytes:Slice(vt2Start+metadata,vt2End)

        if vtableEqual(self.currentVTable, objectOffset, vt2) then
            exisitingVTable = vt2Offset
            break
        end

        i = i - 1
    end

    if not exisitingVTable then
        i = #self.currentVTable
        while i >= 1 do
            local off = 0
            local a = self.currentVTable[i]
            if a and a ~= 0 then
                off = objectOffset - a
            end
            self:PrependVOffsetT(off)

            i = i - 1
        end

        local objectSize = objectOffset - self.objectEnd
        self:PrependVOffsetT(objectSize)

        local vBytes = #self.currentVTable + VtableMetadataFields
        vBytes = vBytes * VOffsetT.bytewidth
        self:PrependVOffsetT(vBytes)

        local objectStart = #self.bytes - objectOffset
        self.bytes:Set(SOffsetT:Pack(self:Offset() - objectOffset),objectStart)

        table.insert(self.vtables, self:Offset())
    else
        local objectStart = #self.bytes - objectOffset
        self.head = objectStart
        self.bytes:Set(SOffsetT:Pack(exisitingVTable - objectOffset),self.head)
    end

    self.currentVTable = nil
    return objectOffset
end

function mt:EndObject()
    assert(self.nested)
    self.nested = false
    return self:WriteVtable()
end

local function growByteBuffer(self, desiredSize)
    local s = #self.bytes
    assert(s < MAX_BUFFER_SIZE, "Flat Buffers cannot grow buffer beyond 2 gigabytes")
    local newsize = s
    repeat
        newsize = math.min(newsize * 2, MAX_BUFFER_SIZE)
        if newsize == 0 then newsize = 1 end
    until newsize > desiredSize

    self.bytes:Grow(newsize)
end

function mt:Head()
    return self.head
end

function mt:Offset()
   return #self.bytes - self.head
end

function mt:Pad(n)
    if n > 0 then
        -- pads are 8-bit, so skip the bytewidth lookup
        local h = self.head - n  -- UInt8
        self.head = h
        self.bytes:Pad(n, h)
    end
end

function mt:Prep(size, additionalBytes)
    if size > self.minalign then
        self.minalign = size
    end

    local h = self.head

    local k = #self.bytes - h + additionalBytes
    local alignsize = ((~k) + 1) & (size - 1) -- getAlignSize(k, size)

    local desiredSize = alignsize + size + additionalBytes

    while self.head < desiredSize do
        local oldBufSize = #self.bytes
        growByteBuffer(self, desiredSize)
        local updatedHead = self.head + #self.bytes - oldBufSize
        self.head = updatedHead
    end

    self:Pad(alignsize)
end

function mt:PrependSOffsetTRelative(off)
    self:Prep(SOffsetT.bytewidth, 0)
    assert(off <= self:Offset(), "Offset arithmetic error")
    local off2 = self:Offset() - off + SOffsetT.bytewidth
    self:Place(off2, SOffsetT)
end

function mt:PrependUOffsetTRelative(off)
    self:Prep(UOffsetT.bytewidth, 0)
    local soffset = self:Offset()
    if off <= soffset then
        local off2 = soffset - off + UOffsetT.bytewidth
        self:Place(off2, UOffsetT)
    else
        error("Offset arithmetic error")
    end
end

function mt:StartVector(elemSize, numElements, alignment)
    assert(not self.nested)
    self.nested = true
    self:Prep(Uint32.bytewidth, elemSize * numElements)
    self:Prep(alignment, elemSize * numElements)
    return self:Offset()
end

function mt:EndVector(vectorNumElements)
    assert(self.nested)
    self.nested = false
    self:Place(vectorNumElements, UOffsetT)
    return self:Offset()
end

function mt:CreateString(s)
    assert(not self.nested)
    self.nested = true

    assert(type(s) == "string")

    self:Prep(UOffsetT.bytewidth, (#s + 1)*Uint8.bytewidth)
    self:Place(0, Uint8)

    local l = #s
    self.head = self.head - l

    self.bytes:Set(s, self.head, self.head + l)

    return self:EndVector(#s)
end

function mt:CreateByteVector(x)
    assert(not self.nested)
    self.nested = true
    self:Prep(UOffsetT.bytewidth, #x*Uint8.bytewidth)

    local l = #x
    self.head = self.head - l

    self.bytes:Set(x, self.head, self.head + l)

    return self:EndVector(#x)
end

function mt:Slot(slotnum)
    assert(self.nested)
    -- n.b. slot number is 0-based
    self.currentVTable[slotnum + 1] = self:Offset()
end

local function finish(self, rootTable, sizePrefix)
    UOffsetT:EnforceNumber(rootTable)
    local prepSize = UOffsetT.bytewidth
    if sizePrefix then
        prepSize = prepSize + Int32.bytewidth
    end

    self:Prep(self.minalign, prepSize)
    self:PrependUOffsetTRelative(rootTable)
    if sizePrefix then
        local size = #self.bytes - self.head
        Int32:EnforceNumber(size)
        self:PrependInt32(size)
    end
    self.finished = true
    return self.head
end

function mt:Finish(rootTable)
    return finish(self, rootTable, false)
end

function mt:FinishSizePrefixed(rootTable)
    return finish(self, rootTable, true)
end

function mt:Prepend(flags, off)
    self:Prep(flags.bytewidth, 0)
    self:Place(off, flags)
end

function mt:PrependSlot(flags, o, x, d)
    flags:EnforceNumber(x)
    flags:EnforceNumber(d)
    if x ~= d then
        self:Prepend(flags, x)
        self:Slot(o)
    end
end

function mt:PrependBoolSlot(...)    self:PrependSlot(Bool, ...) end
function mt:PrependByteSlot(...)    self:PrependSlot(Uint8, ...) end
function mt:PrependUint8Slot(...)   self:PrependSlot(Uint8, ...) end
function mt:PrependUint16Slot(...)  self:PrependSlot(Uint16, ...) end
function mt:PrependUint32Slot(...)  self:PrependSlot(Uint32, ...) end
function mt:PrependUint64Slot(...)  self:PrependSlot(Uint64, ...) end
function mt:PrependInt8Slot(...)    self:PrependSlot(Int8, ...) end
function mt:PrependInt16Slot(...)   self:PrependSlot(Int16, ...) end
function mt:PrependInt32Slot(...)   self:PrependSlot(Int32, ...) end
function mt:PrependInt64Slot(...)   self:PrependSlot(Int64, ...) end
function mt:PrependFloat32Slot(...) self:PrependSlot(Float32, ...) end
function mt:PrependFloat64Slot(...) self:PrependSlot(Float64, ...) end

function mt:PrependUOffsetTRelativeSlot(o,x,d)
    if x~=d then
        self:PrependUOffsetTRelative(x)
        self:Slot(o)
    end
end

function mt:PrependStructSlot(v,x,d)
    UOffsetT:EnforceNumber(d)
    if x~=d then
        UOffsetT:EnforceNumber(x)
        assert(x == self:Offset(), "Tried to write a Struct at an Offset that is different from the current Offset of the Builder.")
        self:Slot(v)
    end
end

function mt:PrependBool(x)      self:Prepend(Bool, x) end
function mt:PrependByte(x)      self:Prepend(Uint8, x) end
function mt:PrependUint8(x)     self:Prepend(Uint8, x) end
function mt:PrependUint16(x)    self:Prepend(Uint16, x) end
function mt:PrependUint32(x)    self:Prepend(Uint32, x) end
function mt:PrependUint64(x)    self:Prepend(Uint64, x) end
function mt:PrependInt8(x)      self:Prepend(Int8, x) end
function mt:PrependInt16(x)     self:Prepend(Int16, x) end
function mt:PrependInt32(x)     self:Prepend(Int32, x) end
function mt:PrependInt64(x)     self:Prepend(Int64, x) end
function mt:PrependFloat32(x)   self:Prepend(Float32, x) end
function mt:PrependFloat64(x)   self:Prepend(Float64, x) end
function mt:PrependVOffsetT(x)  self:Prepend(VOffsetT, x) end

function mt:Place(x, flags)
    local d = flags:EnforceNumberAndPack(x)
    local h = self.head - flags.bytewidth
    self.head = h
    self.bytes:Set(d, h)
end

return m
