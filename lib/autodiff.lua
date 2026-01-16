--[[
    AUTOMATIC DIFFERENTIATION MODULE

    This is the black magic shit.
]]

local pp = require("cc.pretty").pretty_print --- @type function
local matrix2d = require("lib.matrix2d")
local utils = require("lib.utils")

local btest, bor = bit32.btest, bit32.bor

local autodiff = {}

--[[ CONSTANTS ]]

local VAR_FLAG = {      --- @enum ModelVariableFlag
    NONE           = 0, -- `bit32.lshift(0, 0)`
    REQUIRES_GRAD  = 1, -- `bit32.lshift(1, 0)`
    PARAMETER      = bit32.lshift(1, 1),
    INPUT          = bit32.lshift(1, 2),
    OUTPUT         = bit32.lshift(1, 3),
    DESIRED_OUTPUT = bit32.lshift(1, 4),
    COST           = bit32.lshift(1, 5),
}
autodiff.VAR_FLAG = VAR_FLAG
local VAR_OP = { --- @enum ModelVariableOperation
    NULL          = 0000,
    CREATE        = 0001,
    _UNARY_START  = 1000, -- Demarcator
    RELU          = 1001,
    SOFTMAX       = 1002,
    _BINARY_START = 2000, -- Demarcator
    ADD           = 2001,
    SUB           = 2002,
    MATMUL        = 2003,
    CROSS_ENTROPY = 2004,
}

local VAR_MAX_INPUTS = 2

--- @param op ModelVariableOperation
--- @return integer
local function MV_NUM_INPUTS(op)
    if op < VAR_OP._UNARY_START then return 0 end
    if op < VAR_OP._BINARY_START then return 1 end
    return 2
end

--- @param train_images Matrix2d
--- @param train_labels Matrix2d
--- @param test_images Matrix2d
--- @param test_labels Matrix2d
--- @param epochs integer
--- @param batch_size integer
--- @param learning_rate number
--- @param save_dir string
--- @return TrainingContext
function autodiff.training_context(train_images, train_labels, test_images, test_labels,
                                   epochs, batch_size, learning_rate, save_dir)
    --- @class TrainingContext
    local self = {}
    self.train_images = train_images
    self.train_labels = train_labels
    self.test_images = test_images
    self.test_labels = test_labels
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.save_dir = save_dir
    return self
end

--- @return Model
function autodiff.model()
    --- @class Model
    local self = {}
    self.input = nil          --- @type ModelVariable?
    self.output = nil         --- @type ModelVariable?
    self.desired_output = nil --- @type ModelVariable?
    self.cost = nil           --- @type ModelVariable?
    self.forward_program = {} --- @type table<ModelVariable>
    self.cost_program = {}    --- @type table<ModelVariable>
    self.parameters = {}      --- @type table<string, ModelVariable> For saving weights and biases.

    --[[ MODEL VARIABLE FACTORIES ]]

    --- @param rows integer
    --- @param cols integer
    --- @param flags integer
    --- @param name string?
    --- @return ModelVariable
    function self.var_create(rows, cols, flags, name)
        --- @class ModelVariable
        --- @field flags integer
        --- @field val Matrix2d
        --- @field grad Matrix2d?
        --- @field op ModelVariableOperation
        --- @field inputs table<ModelVariable> 0-2 model variables
        local var = {}
        var.flags = flags or VAR_FLAG.NONE
        var.val = matrix2d.fill(0, rows, cols)
        var.op = VAR_OP.CREATE
        var.inputs = {}
        if btest(flags, VAR_FLAG.REQUIRES_GRAD) then
            var.grad = matrix2d.fill(0, rows, cols)
        end

        if btest(flags, VAR_FLAG.PARAMETER) then
            if not name then error("Parameter has not been assigned a name!", 2) end
            if self.parameters[name] then error("Parameter already exists!", 2) end
            self.parameters[name] = var
        end

        if btest(flags, VAR_FLAG.INPUT) then
            if self.input then error("Input already set!", 2) end
            self.input = var
        end
        if btest(flags, VAR_FLAG.OUTPUT) then
            if self.output then error("Output already set!", 2) end
            self.output = var
        end
        if btest(flags, VAR_FLAG.DESIRED_OUTPUT) then
            if self.desired_output then error("Desired output already set!", 2) end
            self.desired_output = var
        end
        if btest(flags, VAR_FLAG.COST) then
            if self.cost then error("Cost already set!", 2) end
            self.cost = var
        end
        return var
    end

    --- @param input ModelVariable
    --- @param rows integer
    --- @param cols integer
    --- @param flags integer?
    --- @param op ModelVariableOperation
    --- @return ModelVariable
    local function _var_unary_impl(input, rows, cols, flags, op)
        flags = flags or VAR_FLAG.NONE
        if btest(input.flags, VAR_FLAG.REQUIRES_GRAD) then
            flags = bor(flags, VAR_FLAG.REQUIRES_GRAD)
        end
        local out = self.var_create(rows, cols, flags, nil)
        out.op = op
        out.inputs[1] = input
        return out
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @param rows integer
    --- @param cols integer
    --- @param flags integer?
    --- @param op ModelVariableOperation
    --- @return ModelVariable
    local function _var_binary_impl(a, b, rows, cols, flags, op)
        flags = flags or VAR_FLAG.NONE
        if btest(a.flags, VAR_FLAG.REQUIRES_GRAD) or
            btest(b.flags, VAR_FLAG.REQUIRES_GRAD)
        then
            flags = bor(flags, VAR_FLAG.REQUIRES_GRAD)
        end
        local out = self.var_create(rows, cols, flags, nil)
        out.op = op
        out.inputs[1] = a
        out.inputs[2] = b
        return out
    end

    --- @param input ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_relu(input, flags)
        return _var_unary_impl(input, input.val.rows, input.val.cols, flags, VAR_OP.RELU)
    end

    --- @param input ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_softmax(input, flags)
        return _var_unary_impl(input, input.val.rows, input.val.cols, flags, VAR_OP.SOFTMAX)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_add(a, b, flags)
        if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
        if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
        return _var_binary_impl(a, b, a.val.rows, a.val.cols, flags, VAR_OP.ADD)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_sub(a, b, flags)
        if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
        if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
        return _var_binary_impl(a, b, a.val.rows, a.val.cols, flags, VAR_OP.SUB)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_matmul(a, b, flags)
        if a.val.cols ~= b.val.rows then error("Column-row mismatch!", 2) end
        return _var_binary_impl(a, b, a.val.rows, b.val.cols, flags, VAR_OP.MATMUL)
    end

    --- @param p ModelVariable
    --- @param q ModelVariable
    --- @param flags integer?
    --- @return ModelVariable
    function self.var_cross_entropy(p, q, flags)
        if p.val.rows ~= q.val.rows then error("Row mismatch!", 2) end
        if p.val.cols ~= q.val.cols then error("Column mismatch!", 2) end
        return _var_binary_impl(p, q, p.val.rows, p.val.cols, flags, VAR_OP.CROSS_ENTROPY)
    end

    --[[ COMPUTATION GRAPH ]]

    --- @param out_var ModelVariable
    --- @return table<ModelVariable>
    local function create_program(out_var)
        local visited = {} --- @type table<ModelVariable, boolean|nil>
        local stack = {}   --- @type table<ModelVariable>
        local out = {}     --- @type table<ModelVariable>

        local insert, remove = table.insert, table.remove

        -- DFS for reverse topological sort
        insert(stack, out_var)
        while #stack > 0 do
            local cur = remove(stack) --- @type ModelVariable
            if visited[cur] then
                insert(out, cur); goto continue
            end
            visited[cur] = true
            insert(stack, cur)
            for i = 1, #cur.inputs do
                local input = cur.inputs[i]
                if not visited[input] then
                    for j = #stack, 1, -1 do
                        if stack[j] == input then
                            remove(stack, j); break
                        end
                    end
                    insert(stack, input)
                end
            end
            ::continue::
        end

        return out
    end

    --- Forward pass
    --- @param program table<ModelVariable>
    local function compute_program(program)
        local unpack = table.unpack
        for i = 1, #program do
            local cur = program[i]          --- @type ModelVariable
            -- Note that they can be nil, but MV_OP check will prevent nil access.
            local a, b = unpack(cur.inputs) --- @type ModelVariable, ModelVariable
            local co = cur.op
            --- @NOTE: Tried using a table out of switch-statement nostalgia. Didn't work!
            if co == VAR_OP.RELU then
                cur.val = a.val:relu()
            elseif co == VAR_OP.SOFTMAX then
                cur.val = a.val:softmax()
            elseif co == VAR_OP.ADD then
                cur.val = a.val:add(b.val)
            elseif co == VAR_OP.SUB then
                cur.val = a.val:sub(b.val)
            elseif co == VAR_OP.MATMUL then
                cur.val = a.val:matmul(b.val)
            elseif co == VAR_OP.CROSS_ENTROPY then
                cur.val = a.val:cross_entropy(b.val)
            end
        end
    end

    --- Backward pass
    --- @param program table<ModelVariable>
    local function compute_program_grads(program)
        for i = 1, #program do
            local cur = program[i] --- @type ModelVariable
            if btest(cur.flags, VAR_FLAG.REQUIRES_GRAD) == false then goto continue end
            if btest(cur.flags, VAR_FLAG.PARAMETER) then goto continue end
            cur.grad = matrix2d.fill(0, cur.grad.rows, cur.grad.cols) -- Clear
            ::continue::
        end

        local last_var = program[#program]
        last_var.grad = matrix2d.fill(1, last_var.grad.rows, last_var.grad.cols)

        local unpack = table.unpack
        for i = #program, 1, -1 do
            local cur = program[i] --- @type ModelVariable

            if btest(cur.flags, VAR_FLAG.REQUIRES_GRAD) == false then goto continue end

            local a, b = unpack(cur.inputs) --- @type ModelVariable, ModelVariable
            local co = cur.op
            local num_inputs = MV_NUM_INPUTS(co)
            if num_inputs == 1 and
                btest(a.flags, VAR_FLAG.REQUIRES_GRAD) == false
            then
                goto continue
            end
            if num_inputs == 2 and
                btest(a.flags, VAR_FLAG.REQUIRES_GRAD) == false and
                btest(b.flags, VAR_FLAG.REQUIRES_GRAD) == false
            then
                goto continue
            end

            if co == VAR_OP.RELU then
                a.grad = a.grad + matrix2d.relu_grad(a.val, cur.grad)
            elseif co == VAR_OP.SOFTMAX then
                a.grad = matrix2d.softmax_grad_vector(cur.val, cur.grad) -- Intentional
            elseif co == VAR_OP.ADD then
                if btest(a.flags, VAR_FLAG.REQUIRES_GRAD) then
                    a.grad = a.grad + cur.grad
                end
                if btest(b.flags, VAR_FLAG.REQUIRES_GRAD) then
                    b.grad = b.grad + cur.grad
                end
            elseif co == VAR_OP.SUB then
                if btest(a.flags, VAR_FLAG.REQUIRES_GRAD) then
                    a.grad = a.grad + cur.grad -- Intentional
                end
                if btest(b.flags, VAR_FLAG.REQUIRES_GRAD) then
                    b.grad = b.grad - cur.grad
                end
            elseif co == VAR_OP.MATMUL then
                if btest(a.flags, VAR_FLAG.REQUIRES_GRAD) then
                    a.grad = a.grad + cur.grad * b.val:transpose()
                end
                if btest(b.flags, VAR_FLAG.REQUIRES_GRAD) then
                    b.grad = b.grad + a.val:transpose() * cur.grad
                end
            elseif co == VAR_OP.CROSS_ENTROPY then
                local p, q = a, b
                local pgn, pqn = matrix2d.cross_entropy_grad(
                    p.val, q.val, cur.grad, p.grad ~= nil, q.grad ~= nil
                )
                if pgn then p.grad = p.grad + pgn end
                if pqn then q.grad = q.grad + pqn end
            end
            ::continue::
        end
    end

    --[[ PROGRESS SAVING ]]

    --- @param path string
    local function write_to_disk(path)
        local file = fs.open(path, "w")
        local km = {}
        for k, v in pairs(self.parameters) do
            km[k] = v.val.vals
        end
        file.write(textutils.serialiseJSON(km))
        file.close()
    end

    --- @param path string
    function self.load_from_disk(path)
        if not fs.exists(path) then error("Save [" .. path .. "] does not exist!") end
        local file = fs.open(path, "r")
        local km = textutils.unserialiseJSON(file.readAll())
        for k, v in pairs(km) do
            if not self.parameters[k] then
                error("Save [" .. path .. "] contains unknown parameter [" .. k .. "]!")
            end
            if #v ~= #self.parameters[k].val.vals then
                error("Save [" .. path .. "] contains mismatched parameter [" .. k .. "]!")
            end
            self.parameters[k].val.vals = v
        end
        file.close()
    end

    --[[ MODEL INTERFACE ]]

    function self.compile()
        if not self.output then error("Model has no output!", 2) end
        if not self.cost then error("Model has no cost!", 2) end
        self.forward_program = create_program(self.output)
        self.cost_program = create_program(self.cost)
    end

    function self.feed_forward()
        compute_program(self.forward_program)
    end

    --- Stochastic Gradient Descent
    --- @param context TrainingContext
    function self.train(context)
        local train_images = context.train_images
        local train_labels = context.train_labels
        local test_images = context.test_images
        local test_labels = context.test_labels

        local num_examples = train_images.rows
        local input_size = train_images.cols
        local output_size = train_labels.cols
        local num_tests = test_images.rows

        local epochs = context.epochs
        local batch_size = context.batch_size
        local learning_rate = context.learning_rate

        --- @NOTE: this is not good
        local num_batches = math.floor(num_examples / batch_size)

        local training_order = {}
        for i = 1, num_examples do
            training_order[i] = i
        end

        local random, copy_range = math.random, utils.copy_range
        local auto_yield = utils.yielder(250, 4000)

        local tw, th = term.getSize()
        local cx, cy = term.getCursorPos()
        local win = window.create(term.current(), cx, cy + 1, tw, th)
        win.write_at = function(self, x, y, str)
            self.setVisible(false)
            self.setCursorPos(x, y); self.write(str)
            self.setVisible(true)
        end

        for epoch = 1, epochs do
            -- Fisher-Yates shuffle
            for i = #training_order, 2, -1 do
                local j = random(i)
                training_order[i], training_order[j] = training_order[j], training_order[i]
            end

            for batch = 1, num_batches do
                for i = 1, #self.cost_program do
                    local cur = self.cost_program[i] --- @type ModelVariable
                    if btest(cur.flags, VAR_FLAG.PARAMETER) then
                        cur.grad = matrix2d.fill(0, cur.grad.rows, cur.grad.cols)
                    end
                end

                local avg_cost = 0
                for i = 1, batch_size do
                    auto_yield()
                    local order_index = (batch - 1) * batch_size + i
                    local index = training_order[order_index]

                    local img_start = (index - 1) * input_size + 1
                    copy_range(self.input.val.vals, train_images.vals,
                        img_start, img_start + input_size - 1
                    )
                    local lbl_start = (index - 1) * output_size + 1
                    copy_range(self.desired_output.val.vals, train_labels.vals,
                        lbl_start, lbl_start + output_size - 1
                    )

                    compute_program(self.cost_program)
                    compute_program_grads(self.cost_program)

                    avg_cost = avg_cost + self.cost.val:sum()
                end

                avg_cost = avg_cost / batch_size

                for i = 1, #self.cost_program do
                    local cur = self.cost_program[i] --- @type ModelVariable
                    if btest(cur.flags, VAR_FLAG.PARAMETER) then
                        cur.grad = cur.grad:scale(learning_rate / batch_size)
                        cur.val  = cur.val:sub(cur.grad)
                    end
                end

                win:write_at(1, 1, string.format("Epoch %d/%d, Batch %d/%d, Avg Cost: %.4f",
                    epoch, epochs, batch, num_batches, avg_cost
                ))
            end

            local num_correct, avg_cost = 0, 0
            for i = 1, num_tests do
                auto_yield()
                local img_start = (i - 1) * input_size + 1
                copy_range(self.input.val.vals, test_images.vals,
                    img_start, img_start + input_size - 1
                )
                local lbl_start = (i - 1) * output_size + 1
                copy_range(self.desired_output.val.vals, test_labels.vals,
                    lbl_start, lbl_start + output_size - 1
                )

                compute_program(self.cost_program)

                avg_cost = avg_cost + self.cost.val:sum()
                if self.output.val:argmax() == self.desired_output.val:argmax() then
                    num_correct = num_correct + 1
                end
            end

            avg_cost = avg_cost / num_tests

            win:write_at(1, 3, string.format("Test %d/%d. Acc: %d/%d (%.1f%%), Avg Cost: %.4f",
                epoch, epochs, num_correct, num_tests, num_correct / num_tests * 100, avg_cost
            ))

            -- write_to_disk(context.save_dir .. "/" .. epoch)
            win:write_at(1, 5, string.format("Saved epoch %d parameters to disk.", epoch))
        end
        win.setCursorPos(1, 7) -- Dumb hack for timed(...)
    end

    return self
end

return autodiff
