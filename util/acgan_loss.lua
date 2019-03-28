require 'torch'
require 'nn'
local acgan_loss = {}


function acgan_loss.lossUpdate(criterion, input, label,weight)
  local reshape_input=nn.View(-1):setNumInputDims(1)
  local reshape_label=nn.View(-1)
  local input_ = reshape_input:forward(input)
  local label_ = reshape_label:forward(label)
  local err = criterion:forward(input_,label_) * weight
  local df = criterion:backward(input_,label_):mul(weight)
  local df_input = reshape_label:backward(input,df)
  return err, df_input
end

return acgan_loss
