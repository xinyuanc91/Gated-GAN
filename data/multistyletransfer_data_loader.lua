--------------------------------------------------------------------------------
-- Subclass of BaseDataLoader that provides data from two datasets.
-- The samples from the datasets are not aligned.
-- The datasets can have different sizes
--------------------------------------------------------------------------------
require 'data.base_data_loader'

local class = require 'class'
data_util = paths.dofile('data_util.lua')

MultiStyleTransferDataLoader = class('MultiStyleTransferDataLoader', 'BaseDataLoader')

function MultiStyleTransferDataLoader:__init(conf)
  BaseDataLoader.__init(self, conf)
  conf = conf or {}
end

function MultiStyleTransferDataLoader:name()
  return 'MultiStyleTransferDataLoader'
end

function MultiStyleTransferDataLoader:Initialize(opt)
  opt.align_data = 0
  self.dataA = data_util.load_dataset('A', opt, opt.input_nc)
  self.dataB = data_util.load_dataset('B', opt, opt.output_nc)
end

-- actually fetches the data
-- |return|: a table of two tables, each corresponding to
-- the batch for dataset A and dataset B
function MultiStyleTransferDataLoader:GetNextBatch()
  local batchA, pathA = self.dataA:getBatch()
  local batchB, labelB, pathB = self.dataB:getClassBatch()
  return batchA, batchB, labelB, pathA, pathB
end

-- returns the size of each dataset
function MultiStyleTransferDataLoader:size(dataset)
  if dataset == 'B' then
    return self.dataB:size()
  end

  -- return the size of the first dataset by default
  return self.dataA:size()
end
