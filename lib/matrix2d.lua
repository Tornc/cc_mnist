--[[
    MATRIX MODULE (2d only, no tensors!)

    NIH syndrome is bad.
]]

local matrix2d = {}

--- @param self Matrix2d
--- @param m Matrix2d
--- @return Matrix2d
local function add(self, m)
    if self.rows ~= m.rows then error("Row mismatch!", 2) end
    if self.cols ~= m.cols then error("Column mismatch!", 2) end

    local nv, sv, mv = {}, self.vals, m.vals
    for i = 1, #sv do
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

    local nv, sv, mv = {}, self.vals, m.vals
    for i = 1, #sv do
        nv[i] = sv[i] - mv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @param m Matrix2d
--- @return Matrix2d
local function matmul(self, m)
    if self.cols ~= m.rows then error("Column-row mismatch!", 2) end

    local result = matrix2d.fill(0, self.rows, m.cols)
    local rv, rc = result.vals, result.cols
    local sv, sc = self.vals, self.cols
    local mv, mc = m.vals, m.cols
    for i = 1, result.rows do
        local ior = (i - 1) * rc -- Precompute offsets to reduce computations.
        local ios = (i - 1) * sc
        for k = 1, sc do         -- The cache locality thing.
            local iom = (k - 1) * mc
            for j = 1, rc do
                local ir = ior + j
                rv[ir] = rv[ir] + sv[ios + k] * mv[iom + j]
            end
        end
    end
    return result
end

--- @param self Matrix2d
--- @param m Matrix2d
--- @return Matrix2d
local function cross_entropy(self, m)
    if self.rows ~= m.rows then error("Row mismatch!", 2) end
    if self.cols ~= m.cols then error("Column mismatch!", 2) end

    local nv, sv, mv = {}, self.vals, m.vals
    local log = math.log
    for i = 1, #sv do
        nv[i] = sv[i] == 0 and 0 or sv[i] * -log(mv[i])
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- Usage:
--- ```lua
--- local total_grad = matrix2d.new(...)
--- local input = matrix2d.new(...)
--- local grad = matrix2d.new(...)
--- total_grad = total_grad + matrix2d.relu_grad(input, gradient)
--- ```
--- @param input Matrix2d Input
--- @param grad Matrix2d Gradient
--- @return Matrix2d
function matrix2d.relu_grad(input, grad)
    if input.rows ~= grad.rows then error("Row mismatch!", 2) end
    if input.cols ~= grad.cols then error("Column mismatch!", 2) end

    local nv, iv, mv = {}, input.vals, grad.vals
    for i = 1, #iv do
        nv[i] = iv[i] > 0 and mv[i] or 0
    end
    return matrix2d.new(nv, input.rows, input.cols)
end

--- @param input Matrix2d Softmax output
--- @param grad Matrix2d Gradient
--- @return Matrix2d output Jacobian multiplied by the gradient
function matrix2d.softmax_grad(input, grad)
    if input.rows ~= 1 and input.cols ~= 1 then error("Not a row/column vector!", 2) end

    local jv, iv = {}, input.vals
    local size = math.max(input.rows, input.cols)
    for i = 1, size do
        for j = 1, size do
            local delta = (i == j) and 1 or 0
            jv[(j - 1) + (i - 1) * size + 1] = iv[i] * (delta - iv[j])
        end
    end
    return matrix2d.new(jv, size, size):matmul(grad)
end

--- This is O(n) compared to `softmax_grad`'s O(n^2), please use this instead.
--- @param input Matrix2d Softmax output vector
--- @param grad Matrix2d Gradient vector
--- @return Matrix2d
function matrix2d.softmax_grad_vector(input, grad)
    if input.rows ~= 1 and input.cols ~= 1 then error("Not a row/column vector!", 2) end
    if grad.rows ~= 1 and grad.cols ~= 1 then error("Not a row/column vector!", 2) end
    if input.rows ~= grad.rows then error("Row mismatch!", 2) end
    if input.cols ~= grad.cols then error("Column mismatch!", 2) end

    local iv, gv = input.vals, grad.vals
    local size = math.max(input.rows, input.cols)

    local dot = 0
    for i = 1, size do
        dot = dot + iv[i] * gv[i]
    end

    local out = {}
    for i = 1, size do
        out[i] = iv[i] * (gv[i] - dot)
    end

    return matrix2d.new(out, input.rows, input.cols)
end

--- @param p Matrix2d
--- @param q Matrix2d
--- @param grad Matrix2d
--- @param do_p boolean Whether you want to calculate `p_grad`
--- @param do_q boolean Whether you want to calculate `q_grad`
--- @return Matrix2d? p_grad
--- @return Matrix2d? q_grad
function matrix2d.cross_entropy_grad(p, q, grad, do_p, do_q)
    if p.rows ~= q.rows or p.rows ~= grad.rows then error("Row mismatch!", 2) end
    if p.cols ~= q.cols or p.cols ~= grad.cols then error("Column mismatch!", 2) end

    local pv, qv, gv = p.vals, q.vals, grad.vals
    local pgv, qgv = {}, {}
    local log = math.log
    if do_p and do_q then
        for i = 1, #pv do -- All input matrices are of same shape anyway.
            pgv[i] = -log(qv[i]) * gv[i]
            qgv[i] = -pv[i] / qv[i] * gv[i]
        end
    else
        if do_p then
            for i = 1, #pv do
                pgv[i] = -log(qv[i]) * gv[i]
            end
        end
        if do_q then
            for i = 1, #pv do
                qgv[i] = -pv[i] / qv[i] * gv[i]
            end
        end
    end
    return
        do_p and matrix2d.new(pgv, p.rows, p.cols) or nil,
        do_q and matrix2d.new(qgv, q.rows, q.cols) or nil
end

--- @param self Matrix2d
--- @param n number
--- @return Matrix2d
local function scale(self, n)
    local nv, sv = {}, self.vals
    for i = 1, #sv do
        nv[i] = sv[i] * n
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function copy(self)
    local nv, sv = {}, self.vals
    for i = 1, #sv do
        nv[i] = sv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function transpose(self)
    local nv, sv, r, c = {}, self.vals, self.rows, self.cols
    local floor = math.floor
    for i = 1, #sv do
        local x = floor((i - 1) / c)
        local y = (i - 1) % c
        nv[y * r + x + 1] = sv[i]
    end
    return matrix2d.new(nv, c, r)
end

--- @param self Matrix2d
--- @return Matrix2d
local function relu(self)
    local nv, sv = {}, self.vals
    local max = math.max
    for i = 1, #sv do
        nv[i] = max(0, sv[i])
    end
    return matrix2d.new(nv, self.rows, self.cols)
end

--- @param self Matrix2d
--- @return Matrix2d
local function softmax(self)
    if self.rows ~= 1 and self.cols ~= 1 then error("Not a row/column vector!", 2) end

    local nv, sv = {}, self.vals
    local sum, exp = 0, math.exp
    for i = 1, #sv do
        nv[i] = exp(sv[i])
        sum = sum + nv[i]
    end
    return matrix2d.new(nv, self.rows, self.cols):scale(1 / sum)
end

--- @param self Matrix2d
--- @return number
local function sum(self)
    if self.rows ~= 1 and self.cols ~= 1 then error("Not a row/column vector!", 2) end

    local _sum, sv = 0, self.vals
    for i = 1, #sv do
        _sum = _sum + sv[i]
    end
    return _sum
end

--- Returns the index with the highest value.
--- @param self Matrix2d Vector
--- @return integer
local function argmax(self)
    if self.rows ~= 1 and self.cols ~= 1 then error("Not a row/column vector!", 2) end

    local mi, sv = 1, self.vals
    for i = 1, #sv do
        if sv[i] > sv[mi] then mi = i end
    end
    return mi
end

--- @param self Matrix2d
--- @return string
local function tostring(self)
    local str = ""
    local sv, c, r = self.vals, self.cols, self.rows
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

--- @param self Matrix2d
--- @param m Matrix2d
--- @param tol number? 1e-6 by default.
--- @return boolean
local function equal(self, m, tol)
    if self.rows ~= m.rows then return false end
    if self.cols ~= m.cols then return false end
    if #self.vals ~= #m.vals then return false end

    tol = tol or 1e-6

    local sv, mv = self.vals, m.vals
    local abs = math.abs
    for i = 1, #sv do
        if abs(sv[i] - mv[i]) > tol then return false end
    end

    return true
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

--- @param lbound number
--- @param ubound number
--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.fill_rand(lbound, ubound, rows, cols)
    local vs, random = {}, math.random
    local range = ubound - lbound
    for i = 1, rows * cols do
        vs[i] = lbound + random() * range
    end
    return matrix2d.new(vs, rows, cols)
end

local metatable = { -- dunder methods!
    __add = add,
    __sub = sub,
    __mul = matmul,
    __tostring = tostring,
    __eq = equal,
}

--- @param values table
--- @param rows integer
--- @param cols integer
--- @return Matrix2d
function matrix2d.new(values, rows, cols)
    if #values ~= rows * cols then error("Table size mismatch!", 2) end

    --- @class Matrix2d
    local self = {}

    self.vals = values
    self.rows = rows -- Row-major btw.
    self.cols = cols

    -- m, m -> m
    self.add = add
    self.sub = sub
    self.matmul = matmul
    self.cross_entropy = cross_entropy

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

    -- m, m, n? -> b
    self.equal = equal

    return setmetatable(self, metatable)
end

return matrix2d
