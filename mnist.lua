--[[ IMPORTS ]]

local autodiff = require("lib.autodiff")
local display = require("lib.display")
local matrix2d = require("lib.matrix2d")
local pretty = require("cc.pretty")
local utils = require("lib.utils")

--[[ CONSTANTS ]]

local TRAINING_PATH = shell.resolve("./dataset/mnist_train.csv")
local TEST_PATH = shell.resolve("./dataset/mnist_test.csv")
local PROGRESS_DIR_PATH = shell.resolve("./progress")

--[[ ALIASES ]]

local VAR_FLAG = autodiff.VAR_FLAG
local auto_yield, timed, copy_range = utils.yielder(1000, 4000), utils.timed, utils.copy_range
local bor = bit32.bor
local pp = pretty.pretty_print --- @type function

--[[ FUNCTIONS ]]

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

--- @return Model model
local function create_mnist_model()
    local model = autodiff.model()

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

    return model
end

--- @param n_epochs integer
--- @param batch_size integer
--- @param learning_rate number
--- @param save string?
--- @param write_to_disk boolean
local function train(n_epochs, batch_size, learning_rate, save, write_to_disk)
    term.clear()
    term.setCursorPos(1, 1)

    if not periphemu then print("You're inside of Minecraft, aren't you?") end
    if not ffi then print("Please consider using LuaJIT.") end

    local y_train, x_train = timed(load_csv, { TRAINING_PATH, 100 }, "Train .csv -> matrix")
    local y_test, x_test = timed(load_csv, { TEST_PATH, nil }, "Test .csv -> matrix")
    local model = create_mnist_model()
    model.compile()
    if save then model.load_from_disk(PROGRESS_DIR_PATH .. "/" .. save) end

    timed(model.train, { autodiff.training_context(
        x_train, y_train, x_test, y_test, n_epochs, batch_size, learning_rate,
        write_to_disk and PROGRESS_DIR_PATH or nil
    ) }, "Training")
end

--- @param save string
--- @param n_samples integer
local function demo(save, n_samples)
    --[[ MODEL SETUP ]]

    local model = create_mnist_model()
    model.compile()
    model.load_from_disk(PROGRESS_DIR_PATH .. "/" .. save)
    local y_test, x_test = load_csv(TEST_PATH, n_samples)

    -- [[ DEMO CONSTANTS ]]

    local WHITE = colours.toBlit(colours.white)
    local LIGHT_GREY = colours.toBlit(colours.lightGrey)
    local GREY = colours.toBlit(colours.grey)
    local BLACK = colours.toBlit(colours.black)
    local RED = colours.toBlit(colours.red)
    local PURPLE = colours.toBlit(colours.purple)

    --[[ DEMO STATE VARIABLES ]]

    local mouse_click = { button = nil, x = nil, y = nil }
    local mouse_drag = { button = nil, x = nil, y = nil }
    local correct_answer = nil

    --- @param x integer
    --- @param y integer
    --- @param content string
    --- @param text_colour string
    --- @param background_colour string
    --- @param action function
    --- @return Button
    local function button(x, y, content, text_colour, background_colour, action)
        --- @class Button
        local self = {}
        self.x0 = x
        self.y0 = y
        self.x1 = x + #content - 1
        self.y1 = y + 1 - 1 --- @NOTE: No multi-line support because it's not needed here.
        self.content = content
        self.text_colour = text_colour
        self.background_colour = background_colour
        self.action = action

        function self.draw(window)
            window.setCursorPos(self.x0, self.y0)
            window.blit(self.content,
                self.text_colour:rep(#self.content),
                self.background_colour:rep(#self.content)
            )
        end

        function self.on_click()
            local _x, _y = mouse_click.x, mouse_click.y
            if not (_x and _y) then return end
            if self.x0 > _x or _x > self.x1 or self.y0 > _y or _y > self.y1 then return end
            self:action() -- No harm in passing self for interesting behaviour.
        end

        return self
    end

    local function input_listener()
        while true do
            local event, p1, p2, p3 = os.pullEvent()
            if event == "mouse_click" then mouse_click = { button = p1, x = p2, y = p3 } end
            if event == "mouse_drag" then mouse_drag = { button = p1, x = p2, y = p3 } end
        end
    end

    local function put(mat, x, y, v)
        local w, h = 28, 28
        if x < 1 or x > w or y < 1 or y > h then return end
        local pos = (y - 1) * w + x
        mat.vals[pos] = math.max(0, math.min(1, mat.vals[pos] + v))
    end

    local function screen()
        term.clear()
        local tw, th = term.getSize()
        local win = window.create(term.current(), 1, 1, tw, th)
        win.write_at = function(self, x, y, str)
            self.setCursorPos(x, y); self.write(str)
        end
        local wx, wy = win.getPosition()

        local cv = display.canvas(28, 30, WHITE)
        local image = matrix2d.fill(0, 784, 1)

        local clear_btn = button(1, 12, "[CLEAR]", WHITE, RED, function()
            image = matrix2d.fill(0, image.rows, image.cols)
            correct_answer = nil
        end)
        local random_btn = button(14, 12, "[RANDOM]", WHITE, PURPLE, function()
            local n = math.random(y_test.rows)

            local y_si = (n - 1) * y_test.cols + 1
            local y_ei = y_si + y_test.cols - 1
            local tmp = {}
            copy_range(tmp, y_test.vals, y_si, y_ei)
            correct_answer = matrix2d.new(tmp, y_test.cols, 1):argmax()

            local x_si = (n - 1) * x_test.cols + 1
            local x_ei = x_si + x_test.cols - 1
            copy_range(image.vals, x_test.vals, x_si, x_ei)
        end)

        while true do
            win.setVisible(false)
            win.clear()
            cv.clear()

            local mb = mouse_click.button or mouse_drag.button
            local mx = mouse_click.x or mouse_drag.x
            local my = mouse_click.y or mouse_drag.y
            if mx and my then
                local rx = (mx - wx + 1) * 2
                local ry = (my - wy + 1) * 3

                -- Either add or subtract the colour value.
                local sign = mb == 1 and 1 or mb == 2 and -1 or 0
                -- Top row
                put(image, rx - 1, ry - 1, 0.50 * sign)
                put(image, rx + 0, ry - 1, 0.75 * sign)
                put(image, rx + 1, ry - 1, 0.50 * sign)
                -- Middle row
                put(image, rx - 1, ry + 0, 0.75 * sign)
                put(image, rx + 0, ry + 0, 1.00 * sign)
                put(image, rx + 1, ry + 0, 0.75 * sign)
                -- Bottom row
                put(image, rx - 1, ry + 1, 0.50 * sign)
                put(image, rx + 0, ry + 1, 0.75 * sign)
                put(image, rx + 1, ry + 1, 0.50 * sign)
            end

            clear_btn.on_click()
            random_btn.on_click()

            if model.input.val ~= image then
                model.input.val = image:copy()
                model.feed_forward()
            end

            -- We don't use the 29th and 30th rows as the images are 28x28.
            cv.fill(1, 29, cv.w, 2, BLACK)
            local px, iv = cv.pixels, image.vals
            for i = 1, #iv do -- Lazy way of displaying.
                if iv[i] >= 0.75 then
                    px[i] = BLACK
                elseif iv[i] >= 0.5 then
                    px[i] = GREY
                elseif iv[i] >= 0.25 then
                    px[i] = LIGHT_GREY
                end
            end

            display.blit_canvas(win, cv)
            clear_btn.draw(win)
            random_btn.draw(win)

            local x_off, y_off = 16, 1
            local i_highlight = model.output.val:argmax()
            local ovv = model.output.val.vals
            for i = 1, #ovv do
                local bgc = colours.black
                if correct_answer then
                    if i == correct_answer then
                        bgc = colours.green
                    elseif i == i_highlight then
                        bgc = (i_highlight == correct_answer) and colours.green or colours.red
                    end
                elseif i == i_highlight then
                    bgc = colours.green
                end

                win.setBackgroundColour(bgc)
                win:write_at(x_off, y_off + i - 1, string.format("%d %3d%%", i - 1, ovv[i] * 100))
                win.setBackgroundColour(colours.black)
            end

            win:write_at(1, 11, "LMB: draw, RMB: erase")
            win.setVisible(true)

            mouse_click, mouse_drag = {}, {}

            os.sleep(0.05)
        end
    end

    parallel.waitForAny(screen, input_listener)
end

if arg[1] == nil or arg[1] == "demo" then
    local save = arg[2]
    if not save then
        local saves = fs.list(PROGRESS_DIR_PATH)
        for i = 1, #saves do
            saves[i] = tonumber(saves[i])
        end
        save = math.max(table.unpack(saves))
    end
    local n_samples = tonumber(arg[3]) or 250
    demo(save, n_samples)
elseif arg[1] == "train" then
    local n_epochs = tonumber(arg[2]) or 20
    local batch_size = tonumber(arg[3]) or 50
    local learning_rate = tonumber(arg[4]) or 0.01
    local save = arg[5] ~= "nil" and arg[5] or nil
    local write_to_disk = arg[6] == "true"

    print("Settings:")
    print(string.format("# epochs      %s", n_epochs))
    print(string.format("batch size    %s", batch_size))
    print(string.format("learning rate %s", learning_rate))
    print(string.format("save          %s", save))
    print(string.format("write to disk %s", write_to_disk))

    print("Continue? y/n")
    term.write("> ")
    local input = io.read()

    if input:lower() == "y" then train(n_epochs, batch_size, learning_rate, save, write_to_disk) end
else
    print("Unknown arguments!")
    print()
    print("Usage:")
    print()
    print("demo [save to load] [# samples]")
    print("      (string)      (integer)")
    print()
    print("train [# epochs] [batch size] [learning rate] [save to load] [write to disk]")
    print("      (integer)   (integer)     (number)       (string?)       (boolean)")
    print("Invalid arguments for 'demo' or 'train' will resort to defaults.")
end
