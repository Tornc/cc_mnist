local pp = require("cc.pretty").pretty_print --- @type function
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local utils = require("lib.utils")

local auto_yield, timed = utils.yielder(), utils.timed

--- @param image table<integer>
local function display_number(image)
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
end

--- @param path string
--- @param max_entries integer?
--- @return table<integer>, table<table<integer>>
local function load_csv(path, max_entries)
    local file = fs.open(path, "r")
    file.readLine(); file.readLine() -- Headers
    local labels, pixels = {}, {}
    for _ = 1, max_entries or 999999999 do
        auto_yield()
        local line = file.readLine()
        if not line then break end
        local itr_cols = string.gmatch(line, "[^,]+")
        table.insert(labels, tonumber(itr_cols()))
        local temp = {}
        for pixel in itr_cols do
            table.insert(temp, tonumber(pixel))
        end
        table.insert(pixels, temp)
    end
    file.close()
    return labels, pixels
end

-- local y_train_raw, x_train_raw = timed(
--     load_csv, { shell.resolve("mnist_train.csv"), 100 }, "Loading training data"
-- )
-- local n = math.random(#x_train_raw)
-- local label, digit = y_train_raw[n], x_train_raw[n]
-- display_number(digit)
-- print(label)

local m1 = matrix2d.new({ 1, 2, 3, 4, 5, 6 }, 2, 3)
local m2 = m1:copy()
print(m1)
m1.values[2] = 0
print(m1, m2)
