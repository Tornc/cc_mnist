--[[
    AUTOMATIC DIFFERENTIATION MODULE

    This is the black magic shit.
]]

local matrix2d = require("lib.matrix2d")

local autodiff = {}

--[[ ENUMS, UTILITY, CLASSES ]]

local MV_FLAG = { --- @enum ModelVariableFlag
    NONE = 1,

    REQUIRES_GRAD = 2,
    PARAMETER = 3,
    INPUT = 4,
    OUTPUT = 5,
    DESIRED_OUTPUT = 6,
    COST = 7,
}
local MV_OP = { --- @enum ModelVariableOperation
    NULL = 1,
    CREATE = 2,

    _UNARY_START = 3,

    RELU = 4,
    SOFTMAX = 5,

    _BINARY_START = 6,

    ADD = 7,
    SUB = 8,
    MATMUL = 9,
    CROSS_ENTROPY = 10,
}

local MV_MAX_INPUTS = 2
local MV_NUM_INPUTS = function(op)
    if op < MV_OP._UNARY_START then return 0 end
    if op < MV_OP._BINARY_START then return 1 end
    return 2
end

--- @TODO: not too much a fan of flag tables anymore...
--- @param flags table<ModelVariableFlag, boolean>
--- @param flag ModelVariableFlag
--- @return table<ModelVariableFlag, boolean>
local add_flag = function(flags, flag)
    local tbl = {}
    for k, _ in pairs(flags) do tbl[k] = true end
    tbl[flag] = true
    return tbl
end

--- @TODO: make some of these params optional.
--- @param index integer
--- @param flags table<ModelVariableFlag, boolean>
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

--- @param num_vars integer
--- @param input ModelVariable
--- @param output ModelVariable
--- @param desired_output ModelVariable
--- @param cost ModelVariable
--- @param forward_prog table<ModelVariable>
--- @param cost_prog table<ModelVariable>
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

--- @param train_images Matrix2d
--- @param train_labels Matrix2d
--- @param test_images Matrix2d
--- @param test_labels Matrix2d
--- @param epochs integer
--- @param batch_size integer
--- @param learning_rate number
--- @return ModelTrainingDescription
local function model_training_desc(train_images, train_labels, test_images, test_labels,
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

--- @param model ModelContext This may get modified.
--- @param rows integer
--- @param cols integer
--- @param flags table<ModelVariableFlag, boolean>
--- @return ModelVariable
local function mv_create(model, rows, cols, flags)
    local out = model_var(
        model.num_vars + 1,
        flags,
        matrix2d.fill(0, rows, cols),
        flags[MV_FLAG.REQUIRES_GRAD] and matrix2d.fill(0, rows, cols) or nil,
        MV_OP.CREATE,
        {}
    )

    --- @TODO: this is not clean ngl.
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
--- @param flags table<ModelVariableFlag, boolean>
--- @param op ModelVariableOperation
--- @return ModelVariable
local function _mv_unary_impl(model, input, rows, cols, flags, op)
    local _flags = flags
    if input.flags[MV_FLAG.REQUIRES_GRAD] then
        _flags = add_flag(flags, MV_FLAG.REQUIRES_GRAD)
    end
    local out = mv_create(model, rows, cols, _flags)
    out.op = op
    out.inputs[1] = input
    return out
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @param rows integer
--- @param cols integer
--- @param flags table<ModelVariableFlag, boolean>
--- @param op ModelVariableOperation
--- @return ModelVariable
local function _mv_binary_impl(model, a, b, rows, cols, flags, op)
    local _flags = flags
    if a.flags[MV_FLAG.REQUIRES_GRAD] or b.flags[MV_FLAG.REQUIRES_GRAD] then
        _flags = add_flag(flags, MV_FLAG.REQUIRES_GRAD)
    end
    local out = mv_create(model, rows, cols, _flags)
    out.op = op
    out.inputs[1] = a
    out.inputs[2] = b
    return out
end

--- @param model ModelContext
--- @param input ModelVariable
--- @param flags table<ModelVariableFlag, boolean>
--- @return ModelVariable
local function mv_relu(model, input, flags)
    return _mv_unary_impl(model, input, input.val.rows, input.val.cols, flags, MV_OP.RELU)
end

--- @param model ModelContext
--- @param input ModelVariable
--- @param flags table<ModelVariableFlag, boolean>
--- @return ModelVariable
local function mv_softmax(model, input, flags)
    return _mv_unary_impl(model, input, input.val.rows, input.val.cols, flags, MV_OP.SOFTMAX)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
local function mv_add(model, a, b, flags)
    if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
    if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, a.val.cols, flags, MV_OP.ADD)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
local function mv_sub(model, a, b, flags)
    if a.val.rows ~= b.val.rows then error("Row mismatch!", 2) end
    if a.val.cols ~= b.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, a.val.cols, flags, MV_OP.SUB)
end

--- @param model ModelContext
--- @param a ModelVariable
--- @param b ModelVariable
--- @return ModelVariable
local function mv_matmul(model, a, b, flags)
    if a.val.cols ~= b.val.rows then error("Column-row mismatch!", 2) end
    return _mv_binary_impl(model, a, b, a.val.rows, b.val.cols, flags, MV_OP.MATMUL)
end

--- @param model ModelContext
--- @param p ModelVariable
--- @param q ModelVariable
--- @return ModelVariable
local function mv_cross_entropy(model, p, q, flags)
    if p.val.rows ~= q.val.rows then error("Row mismatch!", 2) end
    if p.val.cols ~= q.val.cols then error("Column mismatch!", 2) end
    return _mv_binary_impl(model, p, q, p.val.rows, p.val.cols, flags, MV_OP.CROSS_ENTROPY)
end

--- @TODO: what if we made a class and put these functions inside?
--[[ MODEL PROGRAM FUNCTIONS ]]

--- @param model ModelContext
--- @param out_var ModelVariable
--- @return table<ModelVariable>
local function model_program_create(model, out_var)
    return error("Not implemented!", 2)
end

local function model_prog_compute(prog)
    return error("Not implemented!", 2)
end

local function model_prog_compute_grads(prog)
    return error("Not implemented!", 2)
end

--- @TODO: ?????????????????????
--[[ MODEL CONTEXT FUNCTIONS ]]

local function model_create()

end

local function model_compile()

end

local function model_feed_forward()

end

local function model_train()

end


return autodiff
