--[[
    AUTOMATIC DIFFERENTIATION MODULE

    This is the black magic shit.
]]

local matrix2d = require("lib.matrix2d")

local autodiff = {}

--[[ ENUMS, UTILITY, CLASSES ]]

local MV_FLAG = { --- @enum ModelVariableFlag
    NONE           = bit32.lshift(0, 0),
    REQUIRES_GRAD  = bit32.lshift(1, 0),
    PARAMETER      = bit32.lshift(1, 1),
    INPUT          = bit32.lshift(1, 2),
    OUTPUT         = bit32.lshift(1, 3),
    DESIRED_OUTPUT = bit32.lshift(1, 4),
    COST           = bit32.lshift(1, 5),
}
local MV_OP = { --- @enum ModelVariableOperation
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

local MV_MAX_INPUTS = 2

--- @param op ModelVariableOperation
--- @return integer
local MV_NUM_INPUTS = function(op)
    if op < MV_OP._UNARY_START then return 0 end
    if op < MV_OP._BINARY_START then return 1 end
    return 2
end

local btest, bor = bit32.btest, bit32.bor

--- @param index integer
--- @param flags integer
--- @param val Matrix2d
--- @param grad Matrix2d?
--- @param op ModelVariableOperation
--- @param inputs table<ModelVariable>  0-2 model variables
--- @return ModelVariable
local function model_var(index, flags, val, grad, op, inputs)
    if #inputs > MV_MAX_INPUTS then error("Too many inputs!", 2) end

    --- @class ModelVariable
    local self = {}
    self.index = index
    self.flags = flags
    self.val = val
    self.grad = grad
    self.op = op
    self.inputs = inputs
    return self
end

--- @param train_images Matrix2d
--- @param train_labels Matrix2d
--- @param test_images Matrix2d
--- @param test_labels Matrix2d
--- @param epochs integer
--- @param batch_size integer
--- @param learning_rate number
--- @return ModelTrainingDescription
function autodiff.model_training_desc(train_images, train_labels, test_images, test_labels,
                                      epochs, batch_size, learning_rate)
    --- @class ModelTrainingDescription
    local self = {}
    self.train_images = train_images
    self.train_labels = train_labels
    self.test_images = test_images
    self.test_labels = test_labels
    self.epochs = epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    return self
end

--- @return ModelContext
function autodiff.model_context()
    --- @class ModelContext
    local self = {}
    self.num_vars = 0         --- @type integer
    self.input = nil          --- @type ModelVariable?
    self.output = nil         --- @type ModelVariable?
    self.desired_output = nil --- @type ModelVariable?
    self.cost = nil           --- @type ModelVariable?
    self.forward_prog = {}    --- @type table<ModelVariable>
    self.cost_prog = {}       --- @type table<ModelVariable>

    --[[ MODEL VARIABLE FACTORIES ]]

    --- @param rows integer
    --- @param cols integer
    --- @param flags integer
    --- @return ModelVariable
    function self.mv_create(rows, cols, flags)
        local out = model_var(
            self.num_vars,
            flags,
            matrix2d.fill(0, rows, cols),
            btest(flags, MV_FLAG.REQUIRES_GRAD) and matrix2d.fill(0, rows, cols) or nil,
            MV_OP.CREATE,
            {}
        )
        self.num_vars = self.num_vars + 1

        --- @TODO: not really clean ehhh
        if btest(flags, MV_FLAG.INPUT) then
            if self.input then error("Input already set!", 2) end
            self.input = out
        end
        if btest(flags, MV_FLAG.OUTPUT) then
            if self.output then error("Output already set!", 2) end
            self.output = out
        end
        if btest(flags, MV_FLAG.DESIRED_OUTPUT) then
            if self.desired_output then error("Desired output already set!", 2) end
            self.desired_output = out
        end
        if btest(flags, MV_FLAG.COST) then
            if self.cost then error("Cost already set!", 2) end
            self.cost = out
        end
        return out
    end

    --- @param input ModelVariable
    --- @param rows integer
    --- @param cols integer
    --- @param flags integer
    --- @param op ModelVariableOperation
    --- @return ModelVariable
    local function _mv_unary_impl(input, rows, cols, flags, op)
        if btest(input.flags, MV_FLAG.REQUIRES_GRAD) then
            flags = bor(flags, MV_FLAG.REQUIRES_GRAD)
        end
        local out = self.mv_create(rows, cols, flags)
        out.op = op
        out.inputs[1] = input
        return out
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @param rows integer
    --- @param cols integer
    --- @param flags integer
    --- @param op ModelVariableOperation
    --- @return ModelVariable
    local function _mv_binary_impl(a, b, rows, cols, flags, op)
        if btest(a.flags, MV_FLAG.REQUIRES_GRAD) or
            btest(b.flags, MV_FLAG.REQUIRES_GRAD)
        then
            flags = bor(flags, MV_FLAG.REQUIRES_GRAD)
        end
        local out = self.mv_create(rows, cols, flags)
        out.op = op
        out.inputs[1] = a
        out.inputs[2] = b
        return out
    end

    --- @param input ModelVariable
    --- @param flags integer
    --- @return ModelVariable
    function self.mv_relu(input, flags)
        return _mv_unary_impl(input, input.val.rows, input.val.cols, flags, MV_OP.RELU)
    end

    --- @param input ModelVariable
    --- @param flags integer
    --- @return ModelVariable
    function self.mv_softmax(input, flags)
        return _mv_unary_impl(input, input.val.rows, input.val.cols, flags, MV_OP.SOFTMAX)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @return ModelVariable
    function self.mv_add(a, b, flags)
        if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
        if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
        return _mv_binary_impl(a, b, a.val.rows, a.val.cols, flags, MV_OP.ADD)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @return ModelVariable
    function self.mv_sub(a, b, flags)
        if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
        if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
        return _mv_binary_impl(a, b, a.val.rows, a.val.cols, flags, MV_OP.SUB)
    end

    --- @param a ModelVariable
    --- @param b ModelVariable
    --- @return ModelVariable
    function self.mv_matmul(a, b, flags)
        if a.val.cols ~= b.val.rows then error("Column-row mismatch!", 2) end
        return _mv_binary_impl(a, b, a.val.rows, b.val.cols, flags, MV_OP.MATMUL)
    end

    --- @param p ModelVariable
    --- @param q ModelVariable
    --- @return ModelVariable
    function self.mv_cross_entropy(p, q, flags)
        if p.val.rows ~= q.val.rows then error("Row mismatch!", 2) end
        if p.val.cols ~= q.val.cols then error("Column mismatch!", 2) end
        return _mv_binary_impl(p, q, p.val.rows, p.val.cols, flags, MV_OP.CROSS_ENTROPY)
    end

    --[[ COMPUTATION GRAPH ]]

    --- @param out_var ModelVariable
    --- @return table<ModelVariable>
    local function model_prog_create(out_var)
        error("Not implemented!", 2)
        return {}
    end

    --- Forward pass
    --- @param prog table<ModelVariable>
    local function model_prog_compute(prog)
        error("Not implemented!", 2)
        return
    end

    --- Backward pass
    --- @param prog table<ModelVariable>
    local function model_prog_compute_grads(prog)
        error("Not implemented!", 2)
        return
    end

    --[[ MODEL INTERFACE or something ]]

    function self.model_compile()
        if not self.output then error("Model has no output!", 2) end
        if not self.cost then error("Model has no cost!", 2) end
        self.forward_prog = model_prog_create(self.output)
        self.cost_prog = model_prog_create(self.cost)
    end

    function self.model_feed_forward()
        model_prog_compute(self.forward_prog)
    end

    --- @param training_desc ModelTrainingDescription
    function self.model_train(training_desc)
        error("Not implemented!", 2)
    end

    return self
end

return autodiff
