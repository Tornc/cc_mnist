--[[
    YET ANOTHER UTILS MODULE
]]

local utils = {}

--- We need to do this because some functions take longer than 4 seconds.
--- @param check_interval integer amount of times yielder is called before checking the time.
--- @param yield_threshold integer in milliseconds.
--- @return function
function utils.yielder(check_interval, yield_threshold)
    local ci, yt = check_interval, yield_threshold
    local epoch = os.epoch --- @type function
    local ly, i = epoch("utc"), 0
    return function()
        i = i + 1
        if i < ci then return end
        i = 0
        if epoch("utc") - ly < yt then return end
        os.sleep() -- Not sure if localising this is beneficial.
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
