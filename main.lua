-- [[ PLATFORM-SPECIFIC CODE ]]

if not periphemu then print("You're inside of Minecraft, aren't you?") end
if not ffi then print("Please consider using LuaJIT.") end
if periphemu then periphemu.create("front", "monitor") end

--[[ IMPORTS ]]

local autodiff = require("lib.autodiff")
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local pretty = require("cc.pretty")
local utils = require("lib.utils")

--[[ PERIPHERALS ]]

local MONITOR = peripheral.find("monitor")
MONITOR.write_at = function(self, x, y, str) -- I love injection
    self.setCursorPos(x, y); self.write(str)
end

--[[ CONSTANTS ]]

local TRAINING_PATH = shell.resolve("./dataset/mnist_train.csv")
local TEST_PATH = shell.resolve("./dataset/mnist_test.csv")
local PROGRESS_DIR_PATH = shell.resolve("./progress")

local WHITE = colours.toBlit(colours.white)
local LIGHT_GREY = colours.toBlit(colours.lightGrey)
local GREY = colours.toBlit(colours.grey)
local BLACK = colours.toBlit(colours.black)

--[[ ALIASES ]]

local VAR_FLAG = autodiff.VAR_FLAG
local auto_yield, timed, copy_range = utils.yielder(1000, 4000), utils.timed, utils.copy_range
local bor = bit32.bor
local pp = pretty.pretty_print --- @type function

--[[ FUNCTIONS ]]

--- @param image table<number> 0-1
--- @param label table<number> 10 numbers
--- @param pred_pre table<number> 10 numbers
--- @param pred_post table<number>? 10 numbers
local function display_number(image, label, pred_pre, pred_post)
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
--- @return Matrix2d y_matrix One-hot encoded. Shape: n, 10
--- @return Matrix2d x_matrix Normalised to 0-1. Shape: n, 784
local function load_csv(path, max_entries)
    local file = fs.open(path, "r")

    local gmatch = string.gmatch
    local readLine = file.readLine --- @type function

    -- Hardcoded numbers, it's fine because it's a simple application
    local IMAGE_SIZE = 784      -- 28 * 28
    local DIGIT_CATEGORIES = 10 -- 0-9

    local y_vals, x_vals = {}, {}

    readLine(); readLine() -- Skip headers
    for i = 1, max_entries or math.huge do
        auto_yield()
        local line = readLine()
        if not line then break end
        local itr_cols = gmatch(line, "[^,]+")
        y_vals[i] = tonumber(itr_cols())

        local off = (i - 1) * IMAGE_SIZE
        local j = 1
        for pixel in itr_cols do
            x_vals[off + j] = tonumber(pixel)
            j = j + 1
        end
    end
    file.close()

    -- One-hot encoding
    local y_mat = matrix2d.fill(0, #y_vals, DIGIT_CATEGORIES)
    local yv, yc = y_mat.vals, y_mat.cols
    for i = 1, y_mat.rows do
        local digit = y_vals[i] + 1 -- 0-9 -> 1-10 because 1-indexing
        yv[(i - 1) * yc + digit] = 1
    end

    -- Normalise x
    local inv_255 = 1 / 255
    for i = 1, #x_vals do
        x_vals[i] = x_vals[i] * inv_255
    end

    return y_mat, matrix2d.new(x_vals, y_mat.rows, IMAGE_SIZE)
end

--- @param model Model
local function create_mnist_model(model)
    local input = model.var_create(784, 1, VAR_FLAG.INPUT)

    local w0 = model.var_create(16, 784, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "w0")
    local w1 = model.var_create(16, 16, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "w1")
    local w2 = model.var_create(10, 16, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "w2")

    local b0 = model.var_create(16, 1, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "b0")
    local b1 = model.var_create(16, 1, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "b1")
    local b2 = model.var_create(10, 1, bor(VAR_FLAG.REQUIRES_GRAD, VAR_FLAG.PARAMETER), "b2")

    local z0_a = model.var_matmul(w0, input)
    local z0_b = model.var_add(z0_a, b0)
    local a0 = model.var_relu(z0_b)

    local z1_a = model.var_matmul(w1, a0)
    local z1_b = model.var_add(z1_a, b1)
    local z1_c = model.var_relu(z1_b)
    local a1 = model.var_add(a0, z1_c)

    local z2_a = model.var_matmul(w2, a1)
    local z2_b = model.var_add(z2_a, b2)
    local output = model.var_softmax(z2_b, VAR_FLAG.OUTPUT)

    local y = model.var_create(10, 1, VAR_FLAG.DESIRED_OUTPUT)

    local cost = model.var_cross_entropy(y, output, VAR_FLAG.COST)

    -- Initialise weights (not 0)
    local bound0 = math.sqrt(6 / (784 + 16))
    local bound1 = math.sqrt(6 / (16 + 16))
    local bound2 = math.sqrt(6 / (16 + 10))
    w0.val = matrix2d.fill_rand(-bound0, bound0, w0.val.rows, w0.val.cols)
    w1.val = matrix2d.fill_rand(-bound1, bound1, w1.val.rows, w1.val.cols)
    w2.val = matrix2d.fill_rand(-bound2, bound2, w2.val.rows, w2.val.cols)
end

local function main()
    local y_train, x_train = timed(load_csv, { TRAINING_PATH, nil }, "Train .csv -> matrix")
    local y_test, x_test = timed(load_csv, { TEST_PATH, nil }, "Test .csv -> matrix")

    local model = autodiff.model()
    create_mnist_model(model)
    model.compile()

    local n = math.random(y_test.rows)

    local y_si = (n - 1) * y_test.cols + 1
    local y_ei = y_si + y_test.cols - 1
    local label = {}
    copy_range(label, y_test.vals, y_si, y_ei)

    -- Pre-training output
    local x_si = (n - 1) * x_test.cols + 1
    local x_ei = x_si + x_test.cols - 1
    copy_range(model.input.val.vals, x_test.vals, x_si, x_ei)

    model.feed_forward()

    local pred_pre = model.output.val:copy()
    display_number(model.input.val.vals, label, pred_pre.vals)

    timed(model.train, { autodiff.training_context(
        x_train, y_train, x_test, y_test, 1, 50, 0.03, PROGRESS_DIR_PATH
    ) }, "Training")

    -- model.load_from_disk(PROGRESS_DIR_PATH .. "/" .. 3)

    -- Post training output
    copy_range(model.input.val.vals, x_test.vals, x_si, x_ei)
    model.feed_forward()

    local pred_post = model.output.val:copy()
    display_number(model.input.val.vals, label, pred_pre.vals, pred_post.vals)
end

main()

-- pp(fs.list(PROGRESS_DIR_PATH))
