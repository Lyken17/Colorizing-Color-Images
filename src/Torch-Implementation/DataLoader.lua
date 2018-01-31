--[[
  Modified dataloader to load normalized YUV
--]]
require 'torch'
require 'hdf5'
require 'image'
local utils = require 'utils'

local DataLoader = torch.class('DataLoader')


function DataLoader:__init(opt)
  assert(opt.h5_file, 'Must provide h5_file')
  assert(opt.batch_size, 'Must provide batch size')

  self.task = opt.task

  self.h5_file = hdf5.open(opt.h5_file, 'r')
  self.batch_size = opt.batch_size
  
  self.split_idxs = {
    train = 1,
    val = 1,
  }
  
  self.image_paths = {
    train = '/train2014/images',
    val = '/val2014/images',
  }
  
  local train_size = self.h5_file:read(self.image_paths.train):dataspaceSize()
  self.split_sizes = {
    train = train_size[1],
    val = self.h5_file:read(self.image_paths.val):dataspaceSize()[1],
  }
  self.num_channels = train_size[2]
  self.image_height = train_size[3]
  self.image_width = train_size[4]

  self.num_minibatches = {}
  for k, v in pairs(self.split_sizes) do
    self.num_minibatches[k] = math.floor(v / self.batch_size)
  end
  
  if opt.max_train and opt.max_train > 0 then
    self.split_sizes.train = opt.max_train
  end
end

function DataLoader:reset(split)
  self.split_idxs[split] = 1
end

-------------------------------------
------ new part added by Lykenz -----
-------------------------------------


function random(lower, upper)
    local shrink = upper - lower
    return (torch.rand(1) * shrink + lower)[1]
end 

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

function  getLinearMatrix(tmin, tmax)
    local temp = torch.eye(256)
    for ind =1, temp:size()[1] do
        temp[ind][ind] = tmin + (ind - 1) * (tmax - tmin) / (temp:size()[1] - 1)
    end
    return temp:type('torch.FloatTensor')
end

-- linear transform from left to right
linearTmin = 0.6
linearTmax = 1.4
liMat    = getLinearMatrix(linearTmin, linearTmax)
liMatRev = getLinearMatrix(linearTmax, linearTmin)


function linearTrans(img)
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



function DataLoader:getBatch(split)
  local path = self.image_paths[split]

  local start_idx = self.split_idxs[split]
  local end_idx = math.min(start_idx + self.batch_size - 1,
                           self.split_sizes[split])
  
  -- Load images out of the HDF5 file
  local images = self.h5_file:read(path):partial(
                    {start_idx, end_idx},
                    {1, self.num_channels},
                    {1, self.image_height},
                    {1, self.image_width}):float():div(255)

  -- Advance counters, maybe rolling back to the start
  self.split_idxs[split] = end_idx + 1
  if self.split_idxs[split] > self.split_sizes[split] then
    self.split_idxs[split] = 1
  end
  
  --local in_yuv = torch.Tensor(images:size(1),3,self.image_height,self.image_width)
  --local out_uv = torch.Tensor(images:size(1),2,self.image_height,self.image_width)
 
  local tem_shift_img = torch.Tensor(images:size(1), 3, self.image_height,self.image_width)
  local non_shift_img = torch.Tensor(images:size(1), 3, self.image_height,self.image_width)  
    
    
  for t=1,images:size(1) do
    
    local img = images[t]
    
    non_shift_img[t]  = img:clone()
    if random(-1.0, 1.0) <= 0 then
        tem_shift_img[t]  = globalShift(img)
    else 
        tem_shift_img[t]  = linearTrans(img)   
    end
    --[[
    tem_shift_img[t][1] = img[1] * random(0.6, 1.4)
    tem_shift_img[t][2] = img[2]
    tem_shift_img[t][3] = img[3] * random(0.6, 1.4)
        
    for ind = 1, 3 do  
        tem_shift_img[t][ind][torch.lt(tem_shift_img[t][ind], 0)] = 0
        tem_shift_img[t][ind][torch.gt(tem_shift_img[t][ind], 1)] = 1
    end
    ]]--    
    --local tmp_shifted_yuv = image.rgb2yuv(tem_shift_img[t])
    --local non_shifted_yuv = image.rgb2yuv(tem_shift_img[t])
    
    --in_yuv[t] = tmp_shifted_yuv
    --out_uv[t][1] = non_shifted_yuv[2]
    --out_uv[t][2] = non_shifted_yuv[3]
  end
  
  -- input -> output
  return tem_shift_img, non_shift_img
end

