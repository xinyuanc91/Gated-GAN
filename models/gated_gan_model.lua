local class = require 'class'
require 'models.base_model'
require 'util.TotalVariation'
require 'models.multistyle_architectures'
require 'util.image_pool'
util = paths.dofile('../util/util.lua')
acgan_loss = paths.dofile('../util/acgan_loss.lua')
GatedGANModel = class('GatedGANModel', 'BaseModel')


function GatedGANModel:__init(conf)
  BaseModel.__init(self, conf)
  conf = conf or {}
end

function GatedGANModel:model_name()
  return 'GatedGANModel'
end

function GatedGANModel:InitializeStates(use_wgan)
  optimState = {learningRate=opt.lr, beta1=opt.beta1,}
  return optimState
end
-- Defines models and networks
function GatedGANModel:Initialize(opt)
  if opt.test == 0 then
    self.fakeBPool = ImagePool(opt.pool_size)
  end
  -- define tensors
  if opt.test == 0 then  -- allocate tensors for training
    self.real_A = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
    self.real_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
    self.fake_B = torch.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

    self.label_B = torch.Tensor(opt.batchSize,1)
    self.one_hot_label = torch.Tensor(opt.batchSize,opt.n_style+1)
    self.one_hot_map = torch.Tensor(opt.batchSize, opt.n_style, opt.fineSize, opt.fineSize)

    self.autoflag = torch.Tensor(opt.batchSize, opt.n_style+1):fill(0)
    self.autoflag:sub(1,opt.batchSize,1+opt.n_style,1+opt.n_style):fill(1)
    if opt.autoencoder_constrain>0 then
      self.rec_A_AE = torch.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
    end
  end
  -- load/define models
  local use_lsgan = ((opt.use_lsgan ~= nil) and (opt.use_lsgan == 1))
  if not use_lsgan then
    self.criterionGAN = nn.BCECriterion()
  else
    self.criterionGAN = nn.MSECriterion()
  end
  self.criterionACGAN = nn.CrossEntropyCriterion()
  self.criterionRec = nn.AbsCriterion()
  self.criterionEnc = nn.MSECriterion()

  local netG_A, netD_A = nil, nil
    local use_sigmoid = (not use_lsgan)
    netG_A = defineG(opt.input_nc, opt.n_style, opt.output_nc, opt.ngf, opt.which_model_netG, opt.arch)
    print('netG...', netG_A)
    netD_A = defineD(opt.output_nc, opt.ndf, opt.which_model_netD, opt.n_layers_D, use_sigmoid,1,opt.n_style)  -- no sigmoid layer
    print('netD...', netD_A)
  self.netD_A=netD_A
  self.netG_A = netG_A

    -- add tv loss
  if opt.tv_strength>0 then
    self.netG_A = self.netG_A:add(nn.TotalVariation(opt.tv_strength))
  end

  -- define real/fake labels
  if opt.test == 0 then
    local D_A_size = self.netD_A:forward(self.real_B)[1]:size()  -- hack: assume D_size_A = D_size_B
    local D_AC_size = self.netD_A:forward(self.real_B)[2]:size()
    self.class_label_B = torch.Tensor(D_AC_size[1],D_AC_size[2],D_AC_size[3])   -- add by CXY

    self.fake_label_B = torch.Tensor(D_A_size):fill(0.0)
    self.real_label_B = torch.Tensor(D_A_size):fill(0.9) -- no soft smoothing

    self.optimStateD_A = self:InitializeStates()
    self.optimStateG_A = self:InitializeStates()
    self:RefreshParameters(opt)
    if opt.continue_train ==1 then
      local paramG = util.load_param('G_A', opt)
      local paramD = util.load_param('D_A', opt)
      self.parametersG_A:copy(paramG[1])
      self.gradparametersG_A:copy(paramG[2])
      self.parametersD_A:copy(paramD[1]) 
      self.gradparametersD_A:copy(paramD[2])
    end
    print('---------- # Learnable Parameters --------------')
    print(('G_A = %d'):format(self.parametersG_A:size(1)))
    print(('gradG_A = %d'):format(self.gradparametersG_A:size(1)))
    print(('D_A = %d'):format(self.parametersD_A:size(1)))
    print('------------------------------------------------')
  end
end

-- Runs the forward pass of the network and
-- saves the result to member variables of the class
function GatedGANModel:Forward(input, opt)
  if opt.which_direction == 'BtoA' then
  	local temp = input.real_A:clone()
  	input.real_A = input.real_B:clone()
  	input.real_B = temp
  end

  if opt.test == 0 then
    self.real_A:copy(input.real_A)
    self.real_B:copy(input.real_B)
    self.label_B:copy(input.label_B)
    self.one_hot_label:copy(util.label2one_hot_label(self.label_B, self.one_hot_label))
    self.class_label_B:copy(util.label2tensor(self.label_B, self.class_label_B))
  end

  if opt.test == 1 then  -- forward for test
    error('test mode is not completed')
  end
end

-- create closure to evaluate f(X) and df/dX of discriminator
function GatedGANModel:fDx_basic(x, gradParams, netD, netG, real, fake, real_label, fake_label, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()

  -- Real  log(D_A(B))
  local output = netD:forward(real)
  local errD_real = self.criterionGAN:forward(output, real_label)
  local df_do = self.criterionGAN:backward(output, real_label)
  netD:backward(real, df_do)
  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake)
  local errD_fake = self.criterionGAN:forward(output, fake_label)
  local df_do2 = self.criterionGAN:backward(output, fake_label)
  netD:backward(fake, df_do2)
  -- Compute loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, gradParams
end

function GatedGANModel:fDx_kplus(x, gradParams, netD, netG, real, fake, real_label, fake_label, class_label, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  -- Real  log(D_A(B))
  local output = netD:forward(real)
  local errD_real = self.criterionGAN:forward(output[1], real_label)
  local df_do = self.criterionGAN:backward(output[1], real_label)
  local errD_real_class = -1
  local df_do_ac = torch.Tensor(opt.batchSize,df_do:size(3),df_do:size(4),opt.n_style):fill(0.0)
  if opt.n_style>1 then
    errD_real_class,df_do_ac = acgan_loss.lossUpdate(self.criterionACGAN,output[2], class_label, opt.lambda_A)  
  end
  netD:backward(real, {df_do,df_do_ac})
  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake)
  local output_class = output[2]
  output=output[1]
  local errD_fake = self.criterionGAN:forward(output, fake_label)
  local df_do2 = self.criterionGAN:backward(output, fake_label)
  -- local errD_fake_class,df_do2_ac = acgan_loss.lossUpdate(self.criterionACGAN,output_class, class_label, opt.lambda_A)
  netD:backward(fake, {df_do2,df_do_ac*0})
  -- Compute loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, errD_real_class, gradParams
end

function GatedGANModel:fDx_ac(x, gradParams, netD, netG, real, fake, real_label, fake_label, class_label, opt)
  util.BiasZero(netD)
  util.BiasZero(netG)
  gradParams:zero()
  -- Real  log(D_A(B))
  local output = netD:forward(real)
  local errD_real = self.criterionGAN:forward(output[1], real_label)
  local df_do = self.criterionGAN:backward(output[1], real_label)
  local errD_real_class = -1
  local df_do_ac = torch.Tensor(opt.batchSize,df_do:size(3),df_do:size(4),opt.n_style):fill(0.0)
  if opt.n_style>1 then
    errD_real_class,df_do_ac = acgan_loss.lossUpdate(self.criterionACGAN,output[2], class_label, opt.lambda_A)  
  end
  netD:backward(real, {df_do,df_do_ac})
  -- Fake  + log(1 - D_A(G_A(A)))
  output = netD:forward(fake)
  local output_class = output[2]
  output=output[1]
  local errD_fake = self.criterionGAN:forward(output, fake_label)
  local df_do2 = self.criterionGAN:backward(output, fake_label)
  -- local errD_fake_class,df_do2_ac = acgan_loss.lossUpdate(self.criterionACGAN,output_class, class_label, opt.lambda_A)
  netD:backward(fake, {df_do2,df_do_ac*0})
  -- Compute loss
  local errD = (errD_real + errD_fake) / 2.0
  return errD, errD_real_class, gradParams
end


function GatedGANModel:fDAx(x, opt)
  -- use image pool that stores the old fake images
  fake_B = self.fakeBPool:Query(self.fake_B)
  self.errD_A, self.errD_AC, gradParams = self:fDx_ac(x, self.gradparametersD_A, self.netD_A, self.netG_A, self.real_B, fake_B, self.real_label_B, self.fake_label_B, self.class_label_B, opt)
  return self.errD_A, gradParams
end

function GatedGANModel:fGx_ac(x, gradParams, netG, netD_ac, real,real2, real_label, class_label, one_hot_label, opt, netG_encode)
  util.BiasZero(netD_ac)
  util.BiasZero(netG)
  gradParams:zero()

  -- auto-encoder loss
  local rec = netG:forward({real, self.autoflag}):clone()
  local errRec = nil 
  local df_do_rec = nil
  if opt.L2_loss >0 then
    errRec = self.criterionEnc:forward(rec, real)*opt.autoencoder_constrain
    df_do_rec = self.criterionEnc:backward(rec, real):mul(opt.autoencoder_constrain)
  else
    errRec = self.criterionRec:forward(rec, real)*opt.autoencoder_constrain
    df_do_rec = self.criterionRec:backward(rec, real):mul(opt.autoencoder_constrain)
  end
  netG:backward({real, self.autoflag}, df_do_rec)

  --- GAN loss: D_A(G_A(A))
  local fake = netG:forward({real, one_hot_label}):clone()
  local output= netD_ac:forward(fake)
  local errG = self.criterionGAN:forward(output[1], real_label) 
  local df_do1 = self.criterionGAN:backward(output[1], real_label)
  local errG_AC = nil
  local df_do1_ac = torch.Tensor(opt.batchSize,df_do1:size(3),df_do1:size(4),opt.n_style):fill(0.0)
  if opt.n_style>1 then
    errG_AC, df_do1_ac = acgan_loss.lossUpdate(self.criterionACGAN,output[2],class_label, opt.lambda_A)
  end
  local df_d_GAN = netD_ac:updateGradInput(fake, {df_do1,df_do1_ac})

  netG:backward({real, one_hot_label}, df_d_GAN)

  return gradParams, errG, errG_AC, errRec, errI, errEncode, fake, rec
end


function GatedGANModel:fGAx(x, opt)
  self.gradparametersG_A, self.errG_A, self.errG_AC, self.errRec_A, self.errI_B, 
  self.errEncode, self.fake_B,self.rec_A, self.identity_B = 
  self:fGx_ac(x, self.gradparametersG_A, self.netG_A, self.netD_A, self.real_A, self.real_B,
  self.real_label_B, self.class_label_B, self.one_hot_label, opt, self.netG_A_encoder)
  return self.errG_A, self.gradparametersG_A
end                      

function GatedGANModel:OptimizeParameters(opt)
  local fDA = function(x) return self:fDAx(x, opt) end
  local fGA = function(x) return self:fGAx(x, opt) end

  optim.adam(fGA, self.parametersG_A, self.optimStateG_A)
  optim.adam(fDA, self.parametersD_A, self.optimStateD_A)
end

function GatedGANModel:RefreshParameters(opt)
  self.parametersD_A, self.gradparametersD_A = nil, nil -- nil them to avoid spiking memory
  self.parametersG_A, self.gradparametersG_A = nil, nil
  -- define parameters of optimization
  self.parametersG_A, self.gradparametersG_A = self.netG_A:getParameters()
  self.parametersD_A, self.gradparametersD_A = self.netD_A:getParameters()
end

function GatedGANModel:Save(prefix, opt)
  util.save_model(self.netG_A, prefix .. '_net_G.t7', 1)
  util.save_model(self.netD_A, prefix .. '_net_D.t7', 1)
end

function GatedGANModel:SaveParam( prefix, opt )
  util.save_param({self.parametersG_A, self.gradparametersG_A}, prefix ..'_net_G_param.t7', 1)
  util.save_param({self.parametersD_A, self.gradparametersD_A}, prefix ..'_net_D_param.t7', 1)
end

function GatedGANModel:GetCurrentErrorDescription()
  description = ('[A] G: %.4f  G_AC: %.4f  D: %.4f  D_AC: %.4f  Rec: %.4f '):format(
                         self.errG_A and self.errG_A or -1,
                         self.errG_AC and self.errG_AC or -1,
                         self.errD_A and self.errD_A or -1,
                         self.errD_AC and self.errD_AC or -1,
                         self.errRec_A and self.errRec_A or -1)
  return description
end

function GatedGANModel:GetCurrentErrors()
  local errors = {errG_A=self.errG_A, errG_AC=self.errG_AC, errD_A=self.errD_A, 
              errD_AC=self.errD_AC, errRec_A=self.errRec_A, errI_B=self.errI_B}
  return errors
end

-- returns a string that describes the display plot configuration
function GatedGANModel:DisplayPlot(opt)
  return 'errG_A, errG_AC, errD_A, errD_AC, errRec_A'
end

function GatedGANModel:UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD_A['learningRate']
  local lr =  old_lr - lrd
  self.optimStateD_A['learningRate'] = lr
  self.optimStateG_A['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end

local function MakeIm3(im)
  if im:size(2) == 1 then
    local im3 = torch.repeatTensor(im, 1,3,1,1)
    return im3
  else
    return im
  end
end

function GatedGANModel:GetCurrentVisuals(opt, size)
  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.real_A), label='real_A'})
  table.insert(visuals, {img=MakeIm3(self.fake_B), label='fake_B'})
  table.insert(visuals, {img=MakeIm3(self.real_B), label='real_B'})

  if opt.test == 0 and opt.autoencoder_constrain >0 then
    table.insert(visuals, {img=MakeIm3(self.rec_A), label='rec_A_AE'})
  end
  return visuals
end

function GatedGANModel:GetTestResult( opt, test_real_A )
  if opt.gpu >0 then
    self.test_real_A = test_real_A:cuda()
  else
    self.test_real_A = test_real_A:clone()
  end
  self.test_fake_B = {}
  self.map = torch.Tensor(1, opt.n_style, test_real_A:size(3), test_real_A:size(4)):fill(0)
  for i=1,opt.n_style do
    local map = util.label2one_hot_map(torch.Tensor(1,1):fill(i), self.map)
    if opt.gpu > 0 then
      map = map:cuda()
    end
    local fake_B = self.netG_A:forward({self.test_real_A, map}):clone()
    table.insert(self.test_fake_B, fake_B)
  end
  if opt.autoencoder_constrain>0 then
    if opt.gpu > 0 then
      self.map = self.map:cuda()
    end
    local fake_B = self.netG_A:forward({self.test_real_A, self.map:fill(0)}):clone()
    table.insert(self.test_fake_B, fake_B)
  end
end

function GatedGANModel:GetTestVisuals( opt )
  local visuals = {}
  table.insert(visuals, {img=MakeIm3(self.test_real_A), label='test_real_A'})
  for i = 1, opt.n_style do
    local name = (string.format("style %d",i))
    table.insert(visuals, {img=MakeIm3(self.test_fake_B[i]), label=name})
  end
  if opt.autoencoder_constrain>0 then
    table.insert(visuals, {img=MakeIm3(self.test_fake_B[#self.test_fake_B]), label='rec_AE'})
  end
  return visuals
end