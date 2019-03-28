-- code derived from https://github.com/soumith/dcgan.torch and https://github.com/junyanz/CycleGAN

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'


-- load configuration file
options = require 'options'
opt = options.parse_options('train')

require 'models.multistyle_architectures'


-- setup visualization
visualizer = require 'util/visualizer'
-- initialize torch GPU/CPU mode
if opt.gpu > 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu)
  print ("GPU Mode")
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
  print ("CPU Mode")
end

-- load data
-- load data here 'data.multistyletransfer_data_loader'

local data_loader = nil
require 'data.multistyletransfer_data_loader'
data_loader = MultiStyleTransferDataLoader()
print( "DataLoader " .. data_loader:name() .. " was created.")
data_loader:Initialize(opt)

local test_data = torch.Tensor(1,opt.input_nc,opt.loadSize,opt.loadSize)
if opt.valid_freq ~= 0 then
  test_data[1] = util.loadImage(opt.test_data_path, opt.loadSize, 3)
end
-- set batch/instance normalization
set_normalization(opt.norm)
--- timer
local epoch_tm = torch.Timer()
local tm = torch.Timer()

-- define model
local model = nil
local display_plot = nil
if opt.model== 'gated_gan' then
  require 'models.gated_gan_model'
  model = GatedGANModel()
else
  error('Please specify a correct model')
end

-- print the model name
print('Model ' .. model:model_name() .. ' was specified.')
model:Initialize(opt)
-- save initial model
if opt.save_param>0 then
  model:SaveParam('0', opt)
end
-- set up the loss plot
require 'util/plot_util'
plotUtil = PlotUtil()
display_plot = model:DisplayPlot(opt)
plotUtil:Initialize(display_plot, opt.display_id, opt.name)

--------------------------------------------------------------------------------
-- Helper Functions
--------------------------------------------------------------------------------
function visualize_current_results()
  local visuals = model:GetCurrentVisuals(opt)
  for i,visual in ipairs(visuals) do
    visualizer.disp_image(visual.img, opt.display_winsize,
                          opt.display_id+i, opt.name .. ' ' .. visual.label)
  end
end
function visualize_test_results()
  local visuals = model:GetTestVisuals(opt)
  for i,visual in ipairs(visuals) do
    visualizer.disp_image(visual.img, opt.display_winsize,
                          opt.display_id+100+i, opt.name .. '_test_' .. visual.label )
  end
end
function save_current_results(epoch, counter)
  local visuals = model:GetCurrentVisuals(opt)
  visualizer.save_results(visuals, opt, epoch, counter)
end

function save_test_results( epoch, counter )
  local visuals = model:GetTestVisuals(opt)
  visualizer.save_results(visuals, opt, epoch, counter, 'test')
end

function print_current_errors(epoch, counter_in_epoch)
  print(('Epoch: [%d][%8d / %8d]\t Time: %.3f  DataTime: %.3f  '
           .. '%s'):
      format(epoch, ((counter_in_epoch-1) / opt.batchSize),
      math.floor(math.min(data_loader:size('B'), opt.ntrain) / opt.batchSize),
      tm:time().real / opt.batchSize,
      data_loader:time_elapsed_to_fetch_data() / opt.batchSize,
      model:GetCurrentErrorDescription()
  ))
end

function plot_current_errors(epoch, counter_ratio, opt, phase)
  local errs = model:GetCurrentErrors(opt)
  local plot_vals = {epoch + counter_ratio}
  plotUtil:Display(plot_vals, errs)
end

--------------------------------------------------------------------------------
-- Main Training Loop
--------------------------------------------------------------------------------
local counter = 0
local num_batches = math.floor(math.min(data_loader:size('B'), opt.ntrain) / opt.batchSize)
print('#training iterations: ' .. opt.niter+opt.niter_decay )
if opt.start_epoch==nil then
  opt.start_epoch=1
end
for epoch = opt.start_epoch, opt.niter + opt.niter_decay do
    epoch_tm:reset()
    for counter_in_epoch = 1, math.min(data_loader:size('B'), opt.ntrain), opt.batchSize do
        tm:reset()
        -- load a batch and run G on that batch

        local content_image, style_image, style_label, style_path
        content_image, style_image, style_label, _,style_path = data_loader:GetNextBatch()

        if counter == 0 then
           print('sytle_path:',style_path,'style_label:',style_label)
           -- os.exit()
        end
        -- run forward pass
        model:Forward({real_A=content_image,real_B=style_image, label_B=style_label},opt)

        -- run backward pass
        model:OptimizeParameters(opt)

        -- display on the web server
        if counter % opt.display_freq == 0 and opt.display_id > 0 then
          visualize_current_results()
        end

        if counter % opt.valid_freq ==0 and counter >0 then
          model:GetTestResult(opt, test_data)
          visualize_test_results()
        end

        -- logging
        if counter % opt.print_freq == 0 then
          print_current_errors(epoch, counter_in_epoch)
          plot_current_errors(epoch, counter_in_epoch/num_batches, opt)
        end

        -- save latest model
        if counter % opt.save_latest_freq == 0 and counter > 0 then
          print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
          model:Save('latest', opt)
        end

        -- save latest results
        opt.save_display_freq=2500
        if counter % opt.save_display_freq == 0 then
          save_current_results(epoch, counter)
          if opt.valid_freq ~= 0 then
            model:GetTestResult(opt, test_data)
            save_test_results(epoch, counter)
          end
        end
        counter = counter + 1
    end

    -- save model at the end of epoch
    if epoch % opt.save_epoch_freq == 0 then
        print(('saving the model (epoch %d, iters %d)'):format(epoch, counter))
        model:Save('latest', opt)
        model:Save(epoch, opt)
        if opt.save_param>0 then
          model:SaveParam(epoch, opt)
          model:SaveParam('latest',opt)
        end
   end
    -- print the timing information after each epoch
    print(('End of epoch %d / %d \t Time Taken: %.3f'):
        format(epoch, opt.niter + opt.niter_decay, epoch_tm:time().real))

    -- update learning rate
    if epoch > opt.niter then
      model:UpdateLearningRate(opt)
    end
    -- refresh parameters
    model:RefreshParameters(opt)
end
