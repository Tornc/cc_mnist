if not periphemu then print("You're inside of Minecraft, aren't you?") end
if not ffi then print("Please consider using LuaJIT.") end

local pp = require("cc.pretty").pretty_print --- @type function
local autodiff = require("lib.autodiff")
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local utils = require("lib.utils")

local auto_yield, timed = utils.yielder(1000, 4000), utils.timed

local TRAINING_PATH = shell.resolve("./dataset/mnist_train.csv")
local TEST_PATH = shell.resolve("./dataset/mnist_test.csv")

--- @param image table<number> 0-1
--- @param label string
local function display_number(image, label)
    periphemu.create("front", "monitor")
    local MONITOR = peripheral.find("monitor")
    local RED = colours.toBlit(colours.red)
    local WHITE = colours.toBlit(colours.white)
    local LIGHT_GREY = colours.toBlit(colours.lightGrey)
    local GREY = colours.toBlit(colours.grey)
    local BLACK = colours.toBlit(colours.black)

    local cv = display.canvas(28, 30, BLACK)
    local win = window.create(MONITOR, 1, 1, cv.w / 2, cv.h / 3)

    cv.clear()
    local px = cv.pixels
    for i = 1, #image do
        -- This is the lazy way to do it.
        if image[i] > 0.75 then
            px[i] = WHITE
        elseif image[i] > 0.5 then
            px[i] = LIGHT_GREY
        elseif image[i] > 0.25 then
            px[i] = GREY
        end
    end
    display.blit_canvas(win, cv)
    local _, wy = win.getSize()
    MONITOR.setCursorPos(1, wy + 1)
    MONITOR.write(label)
end

--- @param path string
--- @param max_entries integer?
--- @return table<integer>, table<table<integer>>
local function load_csv(path, max_entries)
    local file = fs.open(path, "r")
    local labels, pixels = {}, {}

    local gmatch = string.gmatch
    local readLine = file.readLine --- @type function

    readLine(); readLine()         -- Skip headers
    for _ = 1, max_entries or math.huge do
        local line = readLine()
        if not line then break end
        local itr_cols = gmatch(line, "[^,]+")
        labels[#labels + 1] = tonumber(itr_cols())
        local temp = {}
        for pixel in itr_cols do
            temp[#temp + 1] = tonumber(pixel)
        end
        pixels[#pixels + 1] = temp
    end
    file.close()
    return labels, pixels
end

--- @param y_raw table<integer>
--- @param x_raw table<table<integer>>
--- @return Matrix2d y_matrix One-hot encoded. Shape: n, 10
--- @return Matrix2d x_matrix Normalised to 0-1. Shape: n, 784
local function table_to_matrix(y_raw, x_raw)
    local y_mat = matrix2d.fill(0, #y_raw, 10) -- Yeah, hardcode the 10 different digits.
    local yv = y_mat.vals
    for i = 1, #y_raw do
        local digit = y_raw[i] + 1 -- 0-9 -> 1-10 because 1-indexing
        yv[(i - 1) * 10 + digit] = 1
    end

    local xv = {}
    local xr, xc = #x_raw, #x_raw[1]
    local inv_255 = 1 / 255
    local idx = 1
    for i = 1, xr do
        local tbl = x_raw[i]
        for j = 1, xc do
            xv[idx] = tbl[j] * inv_255
            idx = idx + 1
        end
    end
    return y_mat, matrix2d.new(xv, xr, xc)
end

--- @param m Matrix2d
--- @param n integer row
--- @return table<number> row
local function get_row(m, n)
    local base = (n - 1) * m.cols
    return { table.unpack(m.vals, base + 1, base + m.cols) }
end

local function main()
    local ytr, xtr = timed(
        load_csv, { TRAINING_PATH, 100 }, "Train .csv -> table"
    )
    --- @TODO: we need to shuffle, but; watch out for the fact that y and x are separate though.
    local y_train, x_train = timed(
        table_to_matrix, { ytr, xtr }, "Train table -> matrix"
    )
    local yte, xte = timed(
        load_csv, { TEST_PATH, 1 }, "Test .csv -> table"
    )
    local y_test, x_test = timed(
        table_to_matrix, { yte, xte }, "Test table -> matrix"
    )

    local n = math.random(y_train.rows)
    display_number(get_row(x_train, n), table.concat(get_row(y_train, n), " "))
end

-- main()

-- local mc = autodiff.model_context()

-- local a = mc.mv_create(1, 1, 1)
-- local b = mc.mv_create(1, 1, 2)

-- local params = {"naam"}

-- local function foo1()
--     print("hi!")
-- end
-- local function foo2(n)
--     print("Hello", n)
-- end

-- foo2(table.unpack(params))

-- local smo = matrix2d.fill_rand(-0.5, 0.5, 10000, 1):softmax()
-- local grd = matrix2d.fill_rand(-0.5, 0.5, 10000, 1)

-- local a = timed(matrix2d.softmax_grad,{smo, grd}, "v1")
-- local b = timed(matrix2d.softmax_grad_vector,{smo, grd}, "v2")

-- print(a:equal(b, 1e-6))
