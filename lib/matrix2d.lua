--[[
    MATRIX MODULE

    NIH syndrome is bad.
]]

local pp = require("cc.pretty").pretty_print --- @type function

local matrix2d = {}

--- @param self Matrix2d
--- @param m Matrix2d
--- @return Matrix2d
local function add(self, m)
    if self.rows ~= m.rows then error("Row mismatch!", 2) end
    if self.cols ~= m.cols then error("Column mismatch!", 2) end

    local nv, sv, mv = {}, self.values, m.values
    for i = 1, self.rows * self.cols do
        nv[i] = sv[i] + mv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @param m Matrix2d
--- @return Matrix2d
local function sub(self, m)
    if self.rows ~= m.rows then error("Row mismatch!", 2) end
    if self.cols ~= m.cols then error("Column mismatch!", 2) end

    local nv, sv, mv = {}, self.values, m.values
    for i = 1, self.rows * self.cols do
        nv[i] = sv[i] - mv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

local function mul(self, m)
    if self.rows ~= m.rows then error("Row mismatch!", 2) end
    return error("Not implemented!", 2)
end

--- @param self Matrix2d
--- @param n number
--- @return Matrix2d
local function scale(self, n)
    local nv, sv = {}, self.values
    for i = 1, self.rows * self.cols do
        nv[i] = sv[i] * n
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function copy(self)
    local nv, sv = {}, self.values
    for i = 1, self.rows * self.cols do
        nv[i] = sv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function transpose(self)
    local nv, sv, r, c = {}, self.values, self.rows, self.cols
    local floor = math.floor
    for i = 1, r * c do
        local x = floor((i - 1) / c)
        local y = (i - 1) % c
        nv[y * r + x + 1] = sv[i]
    end
    return matrix2d.new(nv, c, r)
end

--- @param self Matrix2d
--- @return Matrix2d
local function relu(self)
    local nv, sv = {}, self.values
    local max = math.max
    for i = 1, self.rows * self.cols do
        nv[i] = max(0, sv[i])
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function softmax(self)
    local nv, sv = {}, self.values
    local sum, exp = 0, math.exp
    for i = 1, self.rows * self.cols do
        nv[i] = exp(sv[i])
        sum = sum + nv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols):scale(1 / sum)
end

--- @param self Matrix2d
--- @return number
local function sum(self)
    local _sum, sv = 0, self.values
    for i = 1, self.rows * self.cols do
        _sum = _sum + sv[i]
    end
    return _sum
end

--- Returns the index with the highest value.
--- @param self Matrix2d
--- @return integer
local function argmax(self)
    local mi, sv = 1, self.values
    for i = 1, self.rows * self.cols do
        if sv[i] > sv[mi] then mi = i end
    end
    return mi
end

--- @param self Matrix2d
--- @return string
local function tostring(self)
    local str = ""
    local sv, c, r = self.values, self.cols, self.rows
    local off = 0
    for _ = 1, r do
        for x = 1, c do
            str = str .. sv[off + x] .. (x < c and "," or "")
        end
        off = off + c
        str = str .. "\n"
    end
    return str
end

--- @param value number
--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.fill(value, rows, cols)
    local vs = {}
    for i = 1, rows * cols do
        vs[i] = value
    end
    return matrix2d.new(vs, rows, cols)
end

--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.fill_rand(rows, cols)
    local vs = {}
    for i = 1, rows * cols do
        vs[i] = math.random()
    end
    return matrix2d.new(vs, rows, cols)
end

--- @param values table? NOTE: nil acts as uninitialised matrix!
--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.new(values, rows, cols)
    if values and #values ~= rows * cols then error("Table size mismatch!", 2) end

    --- @class Matrix2d
    local self = {}

    self.values = values or {}
    self.rows = rows
    self.cols = cols

    -- m, m -> m
    self.add = add
    self.sub = sub
    self.mul = mul

    -- m, n -> m
    self.scale = scale

    -- m -> m
    self.copy = copy
    self.transpose = transpose
    self.relu = relu
    self.softmax = softmax

    -- m -> v
    self.sum = sum
    self.argmax = argmax
    self.tostring = tostring

    return setmetatable(self, { -- dunder methods!
        __add = add,
        __sub = sub,
        __mul = mul,

        __tostring = tostring,
    })
end

return matrix2d
