local BatchDiscrimination, parent = torch.class('nn.BatchDiscrimination', 'nn.Sequential')

--[[
   Batch discrimination layer, as in "Improved Techniques for Training GANs".
   
   B is ~ "number of kernels"
   C is ~ "kernel dimensions"
   
   Input has to be (batchSize, nInputPlane)
   Then the layer does a matrix multiply (nn.Linear, zero bias) with a nInputPlane*(B*C) matrix -> obtain (batchSize, B*C)
   Then reshape to (batchSize, B, C)
   Then compute L1 distance between rows (collapsing C) -> obtain (batchSize, batchSize, B)

   (**there, authors put 1e6 on the diagonal to obtain 0 on the diagonal after the exp step, I don't**)

   Then x-> exp(-x)
   Then sum over second dimension -> obtain (batchSize, B)
   Then concatenate with input -> obtain (batchSize, nInputPlane + B)

--]]

function BatchDiscrimination:__init(nInputPlane, B, C)
   parent.__init(self)

   assert(nInputPlane and B and C)
   self.B = B
   self.C = C

   local seq = nn.Sequential()
   local l = nn.Linear(nInputPlane, B*C)
   l.bias:zero() 
   seq:add(l)
   seq:add(nn.View(B,C))
   seq:add(nn.L1DistanceBatchMat())
   seq:add(nn.MulConstant(-1, true))
   seq:add(nn.Exp())
   seq:add(nn.Sum(1, 2, true))
   seq:add(nn.Contiguous())

   local concat = nn.Concat(2)
   concat:add(nn.Identity())
   concat:add(seq)

   self:add(concat)

end


function BatchDiscrimination:accGradParameters()
   return
end

if false then
   bs = 5
   nInputPlane = 4
   nkernels = 8
   kerneldim = 3
   m = nn.BatchDiscrimination(nInputPlane,nkernels,kerneldim)
   input = torch.Tensor(bs,nInputPlane):uniform()
   for i=2, bs do
      input[i]:copy(input[1])
   end
   m:forward(input)
   print(m.output)
   m:backward(input, m.output:clone():uniform())
   
   jac = nn.Jacobian
   err=jac.testJacobian(m, input)
   print(err)
   
end

if false then
   bs = 64
   nInputPlane = 1024
   nkernels = 50
   kerneldim = 5
   m = nn.BatchDiscrimination(nInputPlane,nkernels,kerneldim):cuda()
   input = torch.Tensor(bs,nInputPlane):uniform():cuda()
   
   m:forward(input)
   g=m.output:clone():uniform()
   
   for i=1,10 do   
      a = torch.tic()
      m:forward(input)
      print('fwd :', torch.toc(a))
   end
   
   for i=1,100 do   
      a = torch.tic()
      m:backward(input, g)
      print('bwd :', torch.toc(a))
   end
   
end
