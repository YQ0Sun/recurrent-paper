import torch
import torch.nn as nn
import os
import cv2
import torchmetrics
from model import ModelA

## loading the training dataset
train_path='CODaN/data/train'

train_images=[]
train_classes=[]
my_dict={'Bicycle':0,'Boat':1,'Bottle':2,'Bus':3,'Car':4,'Cat':5,'Chair':6,'Cup':7,'Dog':8,'Motorbike':9 }

for classes in os.listdir(train_path):
    
    for img in os.listdir( os.path.join(train_path,classes) ):

        img_path= os.path.join( train_path,classes,img )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        train_images.append(image)
        train_classes.append(my_dict[classes])


train_classes=torch.LongTensor(train_classes)
train_images=torch.FloatTensor(train_images)

## loading the validaiton dataset
val_path='CODaN/data/val'

val_images=[]
val_classes=[]


for classes in os.listdir(val_path):
    
    for img in os.listdir( os.path.join(val_path,classes) ):

        img_path= os.path.join( val_path,classes,img )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        val_images.append(image)
        val_classes.append(my_dict[classes])


val_classes=torch.LongTensor(val_classes)
val_images=torch.FloatTensor(val_images)

# normalizing the dataset
train_images=train_images/255
val_images=val_images/255

# shuffeling the training dataset
num_samples = train_images.size(0)
indices = torch.randperm(num_samples)
train_images = train_images[indices]
train_classes = train_classes[indices]

train_images=train_images.permute(0,3,1,2)  # CNN expects input in the format (batch_size, channels, height, width). 
val_images=val_images.permute(0,3,1,2)

# loading the model
model= ModelA(3,(3,3),1,(2,2),2,4096,10)

# initialising model with random normal weight (as mentioned in paper)
for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.normal_(param, mean=0, std=0.01)
    elif 'bias' in name:
        nn.init.constant_(param, 0)

# optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)



# Training loop


epochs=100
losses=[]
print(f'<<<<<Training started>>>>>')
for epoch in range(epochs):

    preds=model(train_images)
    loss=criterion(preds,train_classes)
    acc=accuracy(preds,train_classes)

    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        preds_val=model(val_images)
        loss_val=criterion(preds_val,val_classes)
        acc_val=accuracy(preds_val,val_classes)

    if epoch%10==0:
        print(f'training loss {loss} val loss {loss_val} training accuracy {acc} val accuracy {acc_val}')
    lr_scheduler.step(acc_val)


