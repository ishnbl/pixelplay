import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

#Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10 
LR = 1e-3
IMAGE_SIZE = (128, 128)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './dataset' 

class Dataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.image_paths = []
        
        split_path = os.path.join(root_dir, split)
        
        video_folders = sorted([f for f in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, f))])

        for vid_id in video_folders:
            vid_path = os.path.join(split_path, vid_id)
            
            files = sorted([f for f in os.listdir(vid_path)])

            #Extract frame numbers using regex
            for filename in files:
                digits = re.findall(r'\d+', filename)
                if not digits:
                    continue
                
                frame_num = int(digits[-1]) 
                full_path = os.path.join(vid_path, filename)
                
                self.image_paths.append((full_path, vid_id, frame_num))

        print(f" -> Found {len(self.image_paths)} images in {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, vid, frame = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', IMAGE_SIZE)
            
        if self.transform:
            image = self.transform(image)
            
        return image, vid, frame

#converts images to tensors
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
#trains the model
def train_model():
    train_dataset = Dataset(DATA_ROOT, split='training_videos', transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = ConvAutoencoder().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for imgs, _, _ in loop:
            imgs = imgs.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
    return model
#runs inference and generates submission file
def generate_submission(model):
    if model is None: return

    test_dataset = Dataset(DATA_ROOT, split='testing_videos', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model.eval()
    criterion = nn.MSELoss(reduction='none') 
    results = []
    
    print("Generating predictions")
    with torch.no_grad():
        for img, vid, frame in tqdm(test_loader, desc="Testing"):
            img = img.to(DEVICE)
            
            output = model(img)
            score = torch.mean(criterion(output, img)).item()
            
            img_flip = torch.flip(img, [2])
            output_flip = model(img_flip)
            score_flip = torch.mean(criterion(output_flip, img_flip)).item()
            
            final_score = min(score, score_flip)
            

            vid_clean = str(int(vid[0])) 
            id_str = f"{vid_clean}_{frame.item()}"
            
            results.append({'Id': id_str, 'Predicted': final_score})
            
    df = pd.DataFrame(results)
    
    df['Predicted'] = (df['Predicted'] - df['Predicted'].min()) / (df['Predicted'].max() - df['Predicted'].min())
    df.to_csv("2d_cnn_images.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    trained_model = train_model()
    generate_submission(trained_model)
