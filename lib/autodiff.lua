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

--- @param index integer
--- @param flags integer
--- @param val Matrix2d
--- @param grad Matrix2d?
--- @param op ModelVariableOperation
--- @param inputs table<ModelVariable|nil>  0-2 model variables
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

--- @TODO: make this into a class
local function model_program2()
end

--- @TODO: make this into a class
--- @param num_vars integer
--- @param input ModelVariable?
--- @param output ModelVariable?
--- @param desired_output ModelVariable?
--- @param cost ModelVariable?
--- @param forward_prog table<ModelVariable|nil>
--- @param cost_prog table<ModelVariable|nil>
--- @return ModelContext
local function model_context(num_vars, input, output, desired_output, cost, forward_prog,
                             cost_prog)
    --- @class ModelContext
    local self = {}
    self.num_vars = num_vars
    self.input = input
    self.output = output
    self.desired_output = desired_output
    self.cost = cost
    self.forward_prog = forward_prog
    self.cost_prog = cost_prog
    return self
end

--- @return ModelContext2
function model_context2()
    --- @class ModelContext2
    local self = {}
    self.num_vars = 0
    self.input = nil          --- @type ModelVariable?
    self.output = nil         --- @type ModelVariable?
    self.desired_output = nil --- @type ModelVariable?
    self.cost = nil           --- @type ModelVariable?
    self.forward_prog = {}    --- @type table<ModelVariable|nil>
    self.cost_prog = {}       --- @type table<ModelVariable|nil>

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

--[[ MODEL VARIABLE FACTORIES ]]

--- @param model ModelContext This will get modified.
--- @param rows integer
--- @param cols integer
--- @param flags integer
--- @return ModelVariable
function autodiff.mv_create(model, rows, cols, flags)
    local out = model_var(
        model.num_vars,
        flags,
        matrix2d.fill(0, rows, cols),
        bit32.btest(flags, MV_FLAG.REQUIRES_GRAD) and matrix2d.fill(0, rows, cols) or nil,
        MV_OP.CREATE,
        {}
    )
    model.num_vars = model.num_vars + 1

    if flags[MV_FLAG.INPUT] then
        if model.input then error("Input already set!", 2) end
        model.input = out
    end
    if flags[MV_FLAG.OUTPUT] then
        if model.output then error("Output already set!", 2) end
        model.output = out
    end
    if flags[MV_FLAG.DESIRED_OUTPUT] then
        if model.desired_output then error("Desired output already set!", 2) end
        model.desired_output = out
    end
    if flags[MV_FLAG.COST] then
        if model.cost then error("Cost already set!", 2) end
        model.cost = out
    end

    return out
end

--- @param model ModelContext
--- @param input ModelVariable
--- @param rows integer
--- @param cols integer
--- @param flags integer
--- @param op ModelVariableOperation
--- @return ModelVariable
local function _mv_unary_impl(model, input, rows, cols, flags, op)
    if bit32.btest(input.flags, MV_FLAG.REQUIRES_GRAD) then
        flags = bit32.bor(flags, MV_FLAG.REQUIRES_GRAD)
    end
    local out = autodiff.mv_create(model, rows, cols, flags)
    out.op = op
    out.inputs[1] = input
    return out
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @param rows integer
--- @param cols integer
--- @param flags integer
--- @param op ModelVariableOperation
--- @return ModelVariable
local function _mv_binary_impl(model, a, b, rows, cols, flags, op)
    if bit32.btest(a.flags, MV_FLAG.REQUIRES_GRAD) or
        bit32.btest(b.flags, MV_FLAG.REQUIRES_GRAD)
    then
        flags = bit32.bor(flags, MV_FLAG.REQUIRES_GRAD)
    end
    local out = autodiff.mv_create(model, rows, cols, flags)
    out.op = op
    out.inputs[1] = a
    out.inputs[2] = b
    return out
end

--- @param model ModelContext
--- @param input ModelVariable
--- @param flags integer
--- @return ModelVariable
function autodiff.mv_relu(model, input, flags)
    return _mv_unary_impl(model, input, input.val.rows, input.val.cols, flags, MV_OP.RELU)
end

--- @param model ModelContext
--- @param input ModelVariable
--- @param flags integer
--- @return ModelVariable
function autodiff.mv_softmax(model, input, flags)
    return _mv_unary_impl(model, input, input.val.rows, input.val.cols, flags, MV_OP.SOFTMAX)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
function autodiff.mv_add(model, a, b, flags)
    if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
    if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, a.val.cols, flags, MV_OP.ADD)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
function autodiff.mv_sub(model, a, b, flags)
    if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
    if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, a.val.cols, flags, MV_OP.SUB)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
function autodiff.mv_matmul(model, a, b, flags)
    if a.val.cols ~= b.val.rows then error("Column-row mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, b.val.cols, flags, MV_OP.MATMUL)
end

--- @param model ModelContext
--- @param p ModelVariable
--- @param q ModelVariable
--- @return ModelVariable
function autodiff.mv_cross_entropy(model, p, q, flags)
    if p.val.rows ~= q.val.rows then error("Row mismatch!", 2) end
    if p.val.cols ~= q.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, p, q, p.val.rows, p.val.cols, flags, MV_OP.CROSS_ENTROPY)
end

--- @TODO: what if we made a class and put these functions inside?
--[[ MODEL PROGRAM FUNCTIONS ]]

--- @param model ModelContext
--- @param out_var ModelVariable
--- @return table<ModelVariable>
local function model_prog_create(model, out_var)
    return error("Not implemented!", 2)
end

local function model_prog_compute(prog)
    return error("Not implemented!", 2)
end

local function model_prog_compute_grads(prog)
    return error("Not implemented!", 2)
end

--- @TODO: This should be inside of a class!
--[[ MODEL CONTEXT FUNCTIONS ]]

--- @return ModelContext
function autodiff.model_create()
    return model_context(
        0, nil, nil, nil, nil, {}, {}
    )
end

--- @param model ModelContext This will get modified.
function autodiff.model_compile(model)
    if not model.output then error("Model has no output!", 2) end
    if not model.cost then error("Model has no cost!", 2) end
    model.forward_prog = model_prog_create(model, model.output)
    model.cost_prog = model_prog_create(model, model.cost)
end

--- @param model ModelContext This will get modified.
local function model_feed_forward(model)
    model_prog_compute(model.forward_prog)
end

--- @param model ModelContext
--- @param training_desc ModelTrainingDescription
local function model_train(model, training_desc)

end


return autodiff
