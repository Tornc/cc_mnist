--[[
    MATRIX MODULE

    NIH syndrome is bad.
]]

local matrix2d = {}

local function add(self, m)
    return error("Not implemented!", 2)
end

local function sub(self, m)
    return error("Not implemented!", 2)
end

local function mul(self, m)
    return error("Not implemented!", 2)
end

local function fill(self, n)
    return error("Not implemented!", 2)
end

local function scale(self, n)
    return error("Not implemented!", 2)
end

local function transpose(self)
    return error("Not implemented!", 2)
end

--- @param self Matrix2d
--- @return Matrix2d
local function copy(self)
    local new_vals, old_vals = {}, self.values
    for i = 1, self.rows * self.cols do
        new_vals[i] = old_vals[i]
    end
    return matrix2d.new(new_vals, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return string
local function tostring(self)
    local str = ""
    local c, r = self.cols, self.rows
    local offset = 0
    for _ = 1, r do
        for x = 1, c do
            str = str .. self.values[offset + x] .. (x < c and "," or "")
        end
        offset = offset + c
        str = str .. "\n"
    end
    return str
end


--- @param values table<number>?
--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.new(values, rows, cols)
    --- @class Matrix2d
    local self = {}

    self.values = values or {}
    if not values then
        for i = 1, rows * cols do
            self.values[i] = 0
        end
    end
    self.rows = rows
    self.cols = cols

    -- m, m
    self.add = add
    self.sub = sub
    self.mul = mul

    -- m, n
    self.fill = fill
    self.scale = scale

    -- m
    self.transpose = transpose
    self.copy = copy

    return setmetatable(self, { -- dunder methods!
        __add = add,
        __sub = sub,
        __mul = mul,

        __tostring = tostring,
    })
end

return matrix2d
