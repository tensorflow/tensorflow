local m = {}

local getAlignSize
if _VERSION == "Lua 5.3" then
    getAlignSize = function(k, size)
            return ((~k) + 1) & (size - 1)
        end    
else
    getAlignSize = function(self, size, additionalBytes)        
        local alignsize = bit32.bnot(#self.bytes-self:Head() + additionalBytes) + 1
        return bit32.band(alignsize,(size - 1))
    end
end
    
m.GetAlignSize = getAlignSize

return m