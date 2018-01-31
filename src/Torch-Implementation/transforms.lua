
require 'torch'
require 'nn'
require 'image'
utils = require 'utils'
require 'nngraph'
require 'ShaveImage'

local transform = {}

local function random(lower, upper)
    local shrink = upper - lower
    return (torch.rand(1) * shrink + lower)[1]
end

transform.random = random

-- global transform for [tmin, tmax]
local linearTmin = 0.6
local linearTmax = 1.4

local function globalShift(img)
    assert(#img:size() == 3)

    local global_shift_img = img:clone()

    global_shift_img[1] = img[1] * random(linearTmin, linearTmax)
    global_shift_img[2] = img[2]
    global_shift_img[3] = img[3] * random(linearTmin, linearTmax)

    -- threash hold, in case range overflow
    for ind = 1, 3 do
        global_shift_img[ind][torch.lt(global_shift_img[ind], 0)] = 0
        global_shift_img[ind][torch.gt(global_shift_img[ind], 1)] = 1
    end
    return global_shift_img
end

transform.globalShift = globalShift

local function  getLinearMatrix(tmin, tmax)
    local temp = torch.eye(256)
    for ind =1, temp:size()[1] do
        temp[ind][ind] = tmin + (ind - 1) * (tmax - tmin) / (temp:size()[1] - 1)
    end
    return temp:type('torch.FloatTensor')
end

transform.getLinearMatrix = getLinearMatrix

-- linear transform from left to right
linearTmin = 0.6
linearTmax = 1.4
liMat    = getLinearMatrix(linearTmin, linearTmax)
liMatRev = getLinearMatrix(linearTmax, linearTmin)


local function linearTrans(img)
    assert(#img:size() == 3)
    if random(-1.0, 1.0) >= 0 then
        mat = liMat
    else
        mat = liMatRev
    end

    local z = img:clone()

    if random(-1.0, 1.0) >= 0 then
        z[1] = img[1] * mat
        z[3] = img[3] * mat
    else
        z[1] = mat * img[1]
        z[3] = mat * img[3]
    end
    for ind = 1,3 do
        z[ind][torch.lt(z[ind], 0)] = 0
        z[ind][torch.gt(z[ind], 1)] = 1
    end
    return z
end

return transform
