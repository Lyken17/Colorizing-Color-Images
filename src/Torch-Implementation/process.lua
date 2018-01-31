-- process Grayball11346 images
-- use do_process_balls.sh for batch processing

require 'torch'
require 'nn'
require 'image'
utils = require 'utils'
require 'nngraph'
require 'ShaveImage'

require 'DataLoader'

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'checkpoint.t7')

-- Input / output options
dataset = "random_image_from_net"
root = "/mnt/HDD1/Datasets/NUS Color/"
cmd:option('-input_dir', root .. dataset .."_disturbed")
cmd:option('-output_dir',root .. dataset .."_recovered")

-- GPU options
cmd:option('-gpu', 0)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)
cmd:option('-cudnn_benchmark', 0)

opt = cmd:parse(arg or {})

dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
print(dtype)
ok, checkpoint = pcall(function() return torch.load(opt.model) end)
-- dtype = 'torch.FloatTensor'
if not ok then
    print('ERROR: Could not load model from ' .. opt.model)
end

model = checkpoint.model
model:evaluate()
model:type(dtype)


-- get file list
function gen_file_list(dir)
    local files = {}

    for file in paths.files(dir) do
        if file:find('[.]jpg$') then
            table.insert(files, file)
        elseif file:find('[.]png$') then
            table.insert(files, file)
        end
    end

    return files
end

function process_file(iFile, oFile)
    local img = image.load(iFile):type('torch.FloatTensor')
    local img_pro = torch.Tensor(img:size())

    local iSize = img:size()
    local input = img:view(1, iSize[1], iSize[2], iSize[3])

    local res = model:forward(input:type(dtype))

    -- input, output comparison
    -- itorch.image({input[1], res[1]})

    -- save output to file
    image.save(oFile, res[1])
end -- function process_file

for i, file in ipairs(gen_file_list(opt.input_dir)) do
    print(file)
    process_file(
        paths.concat(opt.input_dir, file),
        paths.concat(opt.output_dir, file))
end
