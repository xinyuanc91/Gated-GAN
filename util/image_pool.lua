local class = require 'class'
ImagePool= class('ImagePool')
require 'torch'
require 'image'

function ImagePool:__init(pool_size,n_class)
  self.pool_size = pool_size
  if pool_size >0 then
    if n_class==nil then
      self.num_imgs = 0
      self.images = {}
    else
      self.num_imgs = {}
      self.images = {}
      for i = 1,n_class do
        self.num_imgs[i]=0
        table.insert(self.images,{})
      end
    end
  end
end

function ImagePool:model_name()
  return 'ImagePool'
end
-- 
-- function ImagePool:Initialize(pool_size)
--   -- torch.manualSeed(0)
--   -- assert(pool_size > 0)
--   self.pool_size = pool_size
--   if pool_size > 0 then
--     self.num_imgs = 0
--     self.images = {}
--   end
-- end

function ImagePool:Query(image)
  -- print('query image')
  if self.pool_size == 0 then
    -- print('get identical image')
    return image
  end
  if self.num_imgs < self.pool_size then
    -- self.images.insert(image:clone())
    self.num_imgs = self.num_imgs + 1
    self.images[self.num_imgs] = image
    return image
  else
    local p = math.random()
    -- print('p' ,p)
    -- os.exit()
    if p > 0.5 then
      -- print('use old image')
      -- random_id = torch.Tensor(1)
      -- random_id:random(1, self.pool_size)
      local random_id = math.random(self.pool_size)
      -- print('random_id', random_id)
      local tmp = self.images[random_id]:clone()
      self.images[random_id] = image:clone()
      return tmp
    else
      return image
    end

  end
end

function ImagePool:Query_class( image, label )
  -- print ('query Image')
  if self.pool_size == 0 then
    return image
  end
  -- check if image pool need to be filled
  local flag_pool = 0
  for i_sample=1,label:size(1) do 
    local ilabel=label[i_sample][1]
    if self.num_imgs[ilabel] < self.pool_size then
      flag_pool=1
      self.num_imgs[ilabel] =self.num_imgs[ilabel]+1
      self.images[ilabel][self.num_imgs[ilabel]] =image[i_sample]
    end
  end
  if flag_pool ==1 then
    -- return image as pool is not completed
    return image
  else
    local p = math.random()
    if p > 0.5 then
      local tmp = torch.Tensor(image:size())
      local tmplabel = torch.Tensor(label:size())
      for i_sample = 1,image:size(1) do
        local ilabel=label[i_sample][1]
        local random_id = math.random(self.pool_size)
        tmp[i_sample] = self.images[ilabel][random_id]:clone()
        tmplabel[i_sample][1] = ilabel
        self.images[ilabel][random_id] = image[i_sample]:clone()
      end
      return tmp
    else
      return image
    end
  end

end

