--[[
    YET ANOTHER UTILS MODULE
]]

local utils = {}

--- We need to do this because some functions take longer than 4 seconds.
--- @return function
function utils.yielder()
    local ly, i = os.epoch("utc"), 0
    return function()
        i = i + 1
        if i < 1000 then return end
        i = 0
        if os.epoch("utc") - ly < 4000 then return end
        os.sleep()
    end
end

--- @param fn function
--- @param args table
--- @param desc string?
function utils.timed(fn, args, desc)
    local t1 = os.epoch("utc")
    local results = { fn(table.unpack(args)) }
    local str = desc and desc .. " took " or ""
    print(str .. (os.epoch("utc") - t1) .. "ms")
    return table.unpack(results)
end

return utils
