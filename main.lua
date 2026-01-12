if not periphemu then print("You're inside of Minecraft, aren't you?") end
if not ffi then print("Please consider using LuaJIT.") end

local pp = require("cc.pretty").pretty_print --- @type function
local autodiff = require("lib.autodiff")
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local utils = require("lib.utils")

local MV_FLAG = autodiff.MV_FLAG
local auto_yield, timed, copy_range = utils.yielder(1000, 4000), utils.timed, utils.copy_range
local bor = bit32.bor

local TRAINING_PATH = shell.resolve("./dataset/mnist_train.csv")
local TEST_PATH = shell.resolve("./dataset/mnist_test.csv")

--- @param image table<number> 0-1
--- @param label table<number> 10 numbers
--- @param pred_pre table<number> 10 numbers
--- @param pred_post table<number>? 10 numbers
local function display_number(image, label, pred_pre, pred_post)
    periphemu.create("front", "monitor")
    local MONITOR = peripheral.find("monitor")
    MONITOR.write_at = function(self, x, y, str)
        self.setCursorPos(x, y); self.write(str)
    end
    local WHITE = colours.toBlit(colours.white)
    local LIGHT_GREY = colours.toBlit(colours.lightGrey)
    local GREY = colours.toBlit(colours.grey)
    local BLACK = colours.toBlit(colours.black)

    local cv = display.canvas(28, 30, BLACK)
    local win = window.create(MONITOR, 1, 1, cv.w / 2, cv.h / 3)

    MONITOR.clear()
    cv.clear()
    local px = cv.pixels
    for i = 1, #image do -- This is the lazy way to do it.
        if image[i] > 0.75 then
            px[i] = WHITE
        elseif image[i] > 0.5 then
            px[i] = LIGHT_GREY
        elseif image[i] > 0.25 then
            px[i] = GREY
        end
    end
    display.blit_canvas(win, cv)

    local form_digits, form_lbl, form_pre, form_post = {}, {}, {}, {}
    for i = 1, #label do
        form_digits[i] = string.format("%-4s", i - 1)
        form_lbl[i] = string.format("%-4s", label[i])
        form_pre[i] = string.format("%.2f", pred_pre[i])
        if pred_post then form_post[i] = string.format("%.2f", pred_post[i]) end
    end

    local _, wy = win.getSize()
    MONITOR:write_at(1, wy + 1, "D " .. table.concat(form_digits, " ")) -- Digit
    MONITOR:write_at(1, wy + 2, "L " .. table.concat(form_lbl, " "))    -- Label
    MONITOR:write_at(1, wy + 3, "B " .. table.concat(form_pre, " "))    -- Output before
    if not pred_post then return end
    MONITOR:write_at(1, wy + 4, "A " .. table.concat(form_post, " "))   -- Output after
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
        auto_yield()
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

--- @param model ModelContext
local function create_mnist_model(model)
    local input = model.mv_create(784, 1, MV_FLAG.INPUT)

    local w0 = model.mv_create(16, 784, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))
    local w1 = model.mv_create(16, 16, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))
    local w2 = model.mv_create(10, 16, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))

    local bound0 = math.sqrt(6 / (784 + 16))
    local bound1 = math.sqrt(6 / (16 + 16))
    local bound2 = math.sqrt(6 / (16 + 10))
    w0.val = matrix2d.fill_rand(-bound0, bound0, w0.val.rows, w0.val.cols)
    w1.val = matrix2d.fill_rand(-bound1, bound1, w1.val.rows, w1.val.cols)
    w2.val = matrix2d.fill_rand(-bound2, bound2, w2.val.rows, w2.val.cols)

    local b0 = model.mv_create(16, 1, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))
    local b1 = model.mv_create(16, 1, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))
    local b2 = model.mv_create(10, 1, bor(MV_FLAG.REQUIRES_GRAD, MV_FLAG.PARAMETER))

    local z0_a = model.mv_matmul(w0, input)
    local z0_b = model.mv_add(z0_a, b0)
    local a0 = model.mv_relu(z0_b)

    local z1_a = model.mv_matmul(w1, a0)
    local z1_b = model.mv_add(z1_a, b1)
    local z1_c = model.mv_relu(z1_b)
    local a1 = model.mv_add(a0, z1_c)

    local z2_a = model.mv_matmul(w2, a1)
    local z2_b = model.mv_add(z2_a, b2)
    local output = model.mv_softmax(z2_b, MV_FLAG.OUTPUT)

    local y = model.mv_create(10, 1, MV_FLAG.DESIRED_OUTPUT)
    local cost = model.mv_cross_entropy(y, output, MV_FLAG.COST)
end

local function main()
    local ytr, xtr = timed(
        load_csv, { TRAINING_PATH, nil }, "Train .csv -> table"
    )
    local yte, xte = timed(
        load_csv, { TEST_PATH, nil }, "Test .csv -> table"
    )

    local y_train, x_train = timed(
        table_to_matrix, { ytr, xtr }, "Train table -> matrix"
    )
    local y_test, x_test = timed(
        table_to_matrix, { yte, xte }, "Test table -> matrix"
    )

    local model = autodiff.model_context()
    create_mnist_model(model)
    model.model_compile()
    model.model_feed_forward()

    local n = math.random(y_test.rows)

    local x_si = (n - 1) * x_test.cols + 1
    local x_ei = x_si + x_test.cols - 1
    copy_range(model.input.val.vals, x_test.vals, x_si, x_ei)

    local y_si = (n - 1) * y_test.cols + 1
    local y_ei = y_si + y_test.cols - 1
    local label = {}
    copy_range(label, y_test.vals, y_si, y_ei)

    -- Pre-training output
    local pred_pre = model.output.val:copy()
    display_number(model.input.val.vals, label, pred_pre.vals)

    local t1 = os.epoch("utc")
    model.model_train(autodiff.model_training_desc(
        x_train, y_train, x_test, y_test, 1, 50, 0.03
    ))
    local t2 = os.epoch("utc")

    print("Training took: " .. (t2-t1) .. "ms")

    -- Post training output
    local pred_post = model.output.val:copy()
    display_number(model.input.val.vals, label, pred_pre.vals, pred_post.vals)
end

main()
