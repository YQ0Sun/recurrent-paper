import torch
import torch.nn as nn


class ModelA(nn.Module):

    def __init__(self, input_channels : int , conv_kernal_size  , conv_stride: int , maxpool_size  , maxpool_stride : int ,d_ff : int , num_class : int):

        '''
        num_class=3 (for rgb image)
        conv_kernal_size=(3,3) (3x3 as mentioned in the paper)
        conv_stride=1
        maxpool_size=(2,2) (2x2 as mentioned in the paper)
        maxpool_stride=2
        d_ff=4096
        num_classes=10 (according to codan dataset) 

        '''
        super().__init__()
        self.input_channels=input_channels
        self.conv_kernal_size=conv_kernal_size
        self.conv_stride=conv_stride
        self.maxpool_size=maxpool_size
        self.maxpool_stride=maxpool_stride
        self.d_ff=d_ff
        self.num_class=num_class

        self.l0=nn.MaxPool2d(self.maxpool_size,self.maxpool_stride)
        self.dropout=nn.Dropout(0.5)
        self.l1=nn.Conv2d(self.input_channels,64,self.conv_kernal_size,self.conv_stride)
        
        self.l2=nn.Conv2d(64,128,self.conv_kernal_size,self.conv_stride)
        
        self.l3=nn.Conv2d(128,256,self.conv_kernal_size,self.conv_stride)
        self.l4=nn.Conv2d(256,256,self.conv_kernal_size,self.conv_stride)
      
        self.l5=nn.Conv2d(256,512,self.conv_kernal_size,self.conv_stride)
        self.l6=nn.Conv2d(512,512,self.conv_kernal_size,self.conv_stride)
        
        self.l7=nn.Conv2d(512,512,self.conv_kernal_size,self.conv_stride)
        self.l8=nn.Conv2d(512,512,self.conv_kernal_size,self.conv_stride)
        
        self.l9=nn.Linear(3*3*512,self.d_ff)  #(flattening the output by maxpool and feeding to the feed forward layer)
        self.l10=nn.Linear(self.d_ff,self.d_ff)
        self.l11=nn.Linear(self.d_ff,self.num_class)

    def forward(self,x):
        x= self.l1(x)
        x=self.l0(x)

        x=self.l2(x)
        x=self.l0(x)

        x=self.l3(x)
        x=self.l4(x)
        x=self.l0(x)
       
        x=self.l5(x)
        x=self.l6(x)
        x=self.l0(x)

        x=self.l7(x)
        x=self.l8(x)
        x=self.l0(x)
        
        x=torch.reshape(x,(100,3*3*512))
    
        x=self.l9(x)
        x=self.dropout(x)
        x=self.l10(x)
        x=self.dropout(x)
        x=self.l11(x)

        return torch.softmax(x,1)





        


