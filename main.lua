-- https://www.youtube.com/watch?v=hL_n_GljC0I
-- https://github.com/Magicalbat/videos/blob/main/machine-learning/main.c

if not periphemu then print("You're inside of Minecraft, aren't you?") end
if not ffi then print("For the love of god, please use LuaJIT.") end

local pp = require("cc.pretty").pretty_print --- @type function
local autodiff = require("lib.autodiff")
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local utils = require("lib.utils")

local auto_yield, timed = utils.yielder(1000, 4000), utils.timed

--- @param image table<integer>
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
        if image[i] > 192 then
            px[i] = WHITE
        elseif image[i] > 128 then
            px[i] = LIGHT_GREY
        elseif image[i] > 64 then
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
--- @return Matrix2d y_matrix One-hot encoded
--- @return Matrix2d x_matrix A single matrix where 1 row = 1 image
local function table_to_matrix(y_raw, x_raw)
    local y_mat = matrix2d.fill(0, #y_raw, 10) -- Yeah, hardcode the 10 different digits.
    local yv = y_mat.vals
    for i = 1, #y_raw do
        local digit = y_raw[i] + 1 -- 0-9 -> 1-10 because 1-indexing
        yv[(i - 1) * 10 + digit] = 1
    end

    local xv = {}
    local xr, xc = #x_raw, #x_raw[1]
    local idx = 1
    for i = 1, xr do
        local tbl = x_raw[i]
        for j = 1, xc do
            xv[idx] = tbl[j]
            idx = idx + 1
        end
    end
    return y_mat, matrix2d.new(xv, xr, xc)
end

--- @param m Matrix2d
--- @return table<number> row
local function get_row(m, n)
    local base = (n - 1) * m.cols
    return { table.unpack(m.vals, base + 1, base + m.cols) }
end

local ytr, xtr = timed(
    load_csv, { shell.resolve("mnist_train.csv"), 100 }, "Train .csv -> table"
)
local y_train, x_train = timed(
    table_to_matrix, { ytr, xtr }, "Train table -> matrix"
)
-- local yte, xte = timed(
--     load_csv, { shell.resolve("mnist_test.csv"), 1 }, "Test .csv -> table"
-- )
-- local y_test, x_test = timed(
--     table_to_matrix, { yte, xte }, "Test table -> matrix"
-- )

local n = math.random(y_train.rows)
display_number(xtr[n], table.concat(get_row(y_train, n), " ") .. " -> " .. ytr[n])
