import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class Biaffine(nn.Module):
    def __init__(self, in1_features, in2_features, out_features,
                 bias=(True, True), initializer='zeros'):
        super(Biaffine, self).__init__()
        self.initializer = initializer
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        #W = np.zeros((self.linear_output_size, self.linear_input_size), dtype=np.float32)
        #self.linear.weight.data.copy_(torch.from_numpy(W))

        if self.initializer == 'zeros':
            print ("Intializing biaffine linear with zeros")
            nn.init.zeros_(self.linear.weight)
        elif self.initializer == 'orthogonal':
            print ("Intializing biaffine linear with orthogonal")
            nn.init.orthogonal_(self.linear.weight)
        elif self.initializer == 'xavier_uniform':
            print ("Intializing biaffine linear with xavier_uniform")
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = input1.data.new(batch_size, len1, 1).zero_().fill_(1)
            #print(ones)#6*73*1 value=1
            #print(input1) #6*73*500
            input1 = torch.cat((input1, Variable(ones)), dim=2)
            #print(input1) #6*73*501
            dim1 += 1
        if self.bias[1]:
            ones = input2.data.new(batch_size, len2, 1).zero_().fill_(1)
            input2 = torch.cat((input2, Variable(ones)), dim=2)
            dim2 += 1

        affine = self.linear(input1)

        #print("affine before view")
        #print(affine) #compute arc: affine before view is the same as affine after view 6*73*500
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        #print("affine after view")
        #print(affine)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)
        
        #print(biaffine)#compute arc:biaffine before view is 6*73*73, after view is 6*73*73*1
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        #print(biaffine)

        return biaffine

