local m = {} -- the module table

local mt = {} -- the module metatable

-- given a binary array, set a metamethod to return its length
-- (e.g., #binaryArray, calls this)
function mt:__len()
    return self.size
end

-- Create a new binary array of an initial size
function m.New(sizeOrString)
    -- the array storage itself
    local o = {}
    
    if type(sizeOrString) == "string" then
        o.str = sizeOrString
        o.size = #sizeOrString
    elseif type(sizeOrString) == "number" then
        o.data = {}
        o.size = sizeOrString
    else
        error("Expect a integer size value or string to construct a binary array")
    end
    -- set the inheritance
    setmetatable(o, {__index = mt, __len = mt.__len})
    return o
end

-- Get a slice of the binary array from start to end position
function mt:Slice(startPos, endPos)
    startPos = startPos or 0
    endPos = endPos or self.size
    local d = self.data
    if d then
        -- if the self.data is defined, we are building the buffer
        -- in a Lua table
        
        -- new table to store the slice components
        local b = {}
        
        -- starting with the startPos, put all
        -- values into the new table to be concat later
        -- updated the startPos based on the size of the
        -- value
        while startPos < endPos do
            local v = d[startPos] or '/0'
            table.insert(b, v)
            startPos = startPos + #v
        end

        -- combine the table of strings into one string
        -- this is faster than doing a bunch of concats by themselves
        return table.concat(b)
    else
        -- n.b start/endPos are 0-based incoming, so need to convert
        --     correctly. in python a slice includes start -> end - 1
        return self.str:sub(startPos+1, endPos)
    end
end

-- Grow the binary array to a new size, placing the exisiting data
-- at then end of the new array
function mt:Grow(newsize)
    -- the new table to store the data
    local newT = {}
    
    -- the offset to be applied to existing entries
    local offset = newsize - self.size
    
    -- loop over all the current entries and
    -- add them to the new table at the correct
    -- offset location
    local d = self.data
    for i,data in pairs(d) do
        newT[i + offset] = data
    end
    
    -- update this storage with the new table and size
    self.data = newT
    self.size = newsize
end

-- memorization for padding strings
local pads = {}

-- pad the binary with n \0 bytes at the starting position
function mt:Pad(n, startPos)
    -- use memorization to avoid creating a bunch of strings
    -- all the time
    local s = pads[n]
    if not s then
        s = string.rep('\0', n)
        pads[n] = s
    end
    
    -- store the padding string at the start position in the
    -- Lua table
    self.data[startPos] = s
end

-- Sets the binary array value at the specified position
function mt:Set(value, position)
    self.data[position] = value
end

-- locals for slightly faster access
local sunpack = string.unpack
local spack = string.pack

-- Pack the data into a binary representation
function m.Pack(fmt, ...)
    return spack(fmt, ...)
end

-- Unpack the data from a binary representation in
-- a Lua value
function m.Unpack(fmt, s, pos)
    return sunpack(fmt, s.str, pos + 1)
end

-- Return the binary array module
return m