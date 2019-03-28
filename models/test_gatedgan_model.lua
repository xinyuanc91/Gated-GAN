local class = require 'class'
require 'models.base_model'
require 'models.multistyle_architectures'
require 'util.image_pool'
require 'util.InstanceNormalization'
util = paths.dofile('../util/util.lua')
TestGatedGANModel = class('TestGatedGANModel', 'BaseModel')
function TestGatedGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function TestGatedGANModel:model_name()
  return 'TestGatedGANModel'
end

-- Defines models and networks
function TestGatedGANModel:Initialize(opt)
  -- define tensors
  self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
  self.label = {}
  for i=1,opt.n_style do 
    table.insert(self.label, torch.Tensor(opt.batchSize,1):fill(i))
  end

  -- load/define models
  local net_loaded = util.load_test_model('G_A', opt)
  -- setup network
  netG_A = defineG(opt.input_nc, opt.n_style, opt.output_nc, opt.ngf, opt.which_model_netG, opt.arch)
  self.netG_A=netG_A
  -- copy parameter
  self:PassParameters(opt, net_loaded)
  self:RefreshParameters()
end

function select_respart( netG,opt )
  for i=1,opt.n_style do
    netG:get(13):get(2):get(i):get(1):remove(2)
    netG:get(13):get(2):get(i):get(1):add(nn.SelectTable(1))
  end
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function TestGatedGANModel:Forward(input, opt, img_id)
  if opt.which_direction == 'BtoA' then
  	input.real_A = input.real_B:clone()
  end
  self.real_A = input.real_A:clone()
  if opt.gpu > 0 then
    self.real_A = self.real_A:cuda()
  end
  self.gated=torch.Tensor(opt.batchSize, opt.n_style + 1):fill(0.0)
  if opt.gpu>0 then
    self.gated = self.gated:cuda()
  end
  self.fake_B = {}

  for i=1, opt.n_style do
    local gated = util.label2one_hot_label(self.label[i], self.gated)
    if opt.gpu > 0 then
      gated = gated:cuda()
    end
    local fake_B = nil
    fake_B = self.netG_A:forward({self.real_A, gated}):clone()

    table.insert(self.fake_B, fake_B)
end

function TestGatedGANModel:PassParameters(opt,net_loaded)
  for i=1,12 do
    local params_pre, gradParams_pre = net_loaded:get(i):getParameters()
    local params, gradParams = self.netG_A:get(i):getParameters()
    params:copy(params_pre)
  end
  for i=1,opt.n_style do
    local params_pre, gradParams_pre = net_loaded:get(13):get(2):get(i):getParameters()
    local params, gradParams = self.netG_A:get(13):get(2):get(i):getParameters()
    params:copy(params_pre)
  end
  for i=15,28 do
    local params_pre, gradParams_pre = net_loaded:get(i):getParameters()
    local params, gradParams = self.netG_A:get(i):getParameters()
    params:copy(params_pre)
  end
end

function TestGatedGANModel:LoadParameters( opt )
  self.parametersG_A,self.gradparametersG_A = self.netG_A:getParameters()
  local params=torch.load(paths.concat(opt.checkpoints_dir, opt.name,
                                    opt.which_epoch .. '_net_G_param.t7'))
  print(params[1]:size())
  print(self.parametersG_A:size())
  self.parametersG_A:copy(params[1])
  self.gradparametersG_A:copy(params[2])
end

function TestGatedGANModel:RefreshParameters()
  self.parametersG_A, self.gradparametersG_A = nil, nil
  self.parametersG_A, self.gradparametersG_A = self.netG_A:getParameters()
end

local function MakeIm3(im)
  if im:size(2) == 1 then
    local im3 = torch.repeatTensor(im, 1,3,1,1)
    return im3
  else
    return im
  end
end

function TestGatedGANModel:GetCurrentVisuals(opt, size)
  if not size then
    size = opt.display_winsize
  end
  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  -- 
  for i = 1, opt.n_style do
    local name = (string.format("style %d",i))
    local Im = MakeIm3(self.fake_B[i])
    table.insert(visuals, {img=Im, label=name})
  end
  return visuals
end
