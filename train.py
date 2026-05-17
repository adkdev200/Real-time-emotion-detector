# Generated from: train.ipynb
# Converted at: 2026-05-16T11:17:22.092Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.vgg import vgg16
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torchvision.io import read_image
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
import cv2

from torchvision.models.vision_transformer import vit_b_16

model = vit_b_16(weights = True)
model.heads = nn.Sequential(
    nn.Linear(768, 64),
    nn.Linear(64, 6)
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = model
transform = transforms.Compose([
     transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model():
    return model

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image)
    image = image.to(device).unsqueeze(0)
    return image



if __name__ == 'main':
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sujaykapadnis/emotion-recognition-dataset")

    print("Path to dataset files:", path)

    dataset_path = path +'/dataset'
    df_path = path +'/data.csv'
    df = pd.read_csv(df_path)

    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    class EmotionDataset(Dataset):
        def __init__(self, df, df_path, transform):
            self.df = df 
            self.images  = df['path'].tolist()
            self.labels = df['label'].tolist()
            self.df_path = df_path
            self.transform = transform

        def __len__(self):
            return df.shape[0]
        
        def __getitem__(self, idx):
            
            image = Image.open(self.df_path + '/' + self.images[idx])

            label = self.labels[idx]
            image = self.transform(image)
            label = torch.tensor(label, dtype = torch.long)

            return image, label


    emotion_dataset = EmotionDataset(df, dataset_path, transform)

    for p in vgg_model.features.parameters():
        p.requires_grad = True

    dataloader = DataLoader(
        emotion_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    lossfn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(vgg_model.parameters(), lr = 1e-6)

    epochs = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg_model = vgg_model.to(device)
    print(f"Using device : {device}")

    for epoch in range(1):
        sum_loss = 0
        for image, label in tqdm(dataloader):
            image, label = image.to(device), label.to(device)
            with torch.no_grad():
                features = vgg_model.features(image)
            preds = vgg_model.classifier(torch.flatten(features, 1))
            loss = lossfn(preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        print(f"Total Loss : {sum_loss/(len(dataloader))}")

    torch.save(vgg_model.state_dict(), 'emotion_classifier.pt')