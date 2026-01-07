import os
import re
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

#hyperparamaters
CONFIG = {
    'TRAIN_DIR': 'dataset/training_videos',
    'TEST_DIR': 'dataset/testing_videos',
    'SUBMISSION_FILE': 'submission_3d.csv',
    'CLIP_LEN': 16,           
    'IMG_SIZE': 128,
    'BATCH_SIZE': 8,         
    'EPOCHS': 5,
    'LR': 1e-4,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_WORKERS': 4
}




# Transforms to tensor with gaussian blur
common_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])),
    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)), 
    transforms.ToTensor(),
])

class Dataset(Dataset):
    def __init__(self, root_dir, transform=None, clip_len=16, is_train=True):
        self.transform = transform
        self.clip_len = clip_len
        self.clips = []



        video_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
        

        for vid_id in video_folders:
            vid_path = os.path.join(root_dir, vid_id)
            
            files = sorted([
                os.path.join(vid_path, f) for f in os.listdir(vid_path) 
                if f.lower().endswith(('.jpg', '.jpeg'))
            ])
            
            # Sort by frame number
            files.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]))

            if len(files) < clip_len:
                continue

            #use stride for efficiency
            if is_train:
                stride = 4
                for i in range(0, len(files) - clip_len + 1, stride):
                    self.clips.append(files[i : i + clip_len])

        print(f" -> Created {len(self.clips)} clips (Train={is_train})")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        frame_paths = self.clips[idx]
        processed_frames = []
        
        for path in frame_paths:
            try:
                img = Image.open(path).convert('RGB')
            except:
                img = Image.new('RGB', (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])) 
            
            if self.transform:
                img = self.transform(img)
            processed_frames.append(img)
        
        # stack the frames
        return torch.stack(processed_frames, dim=1)



class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),   # (B, 32, 16, 128, 128)
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2)),          # (B, 32, 8, 64, 64)
            
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2)),          # (B, 64, 4, 32, 32)
            
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool3d((2, 2, 2))           # (B, 128, 2, 16, 16)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            nn.ReLU(True),
            
            nn.ConvTranspose3d(32, 3, 2, stride=2),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#training
def train():
    dataset = Dataset(CONFIG['TRAIN_DIR'], transform=common_transforms, is_train=True)
    loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    
    model = Conv3DAutoencoder().to(CONFIG['DEVICE'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    print("Starting training...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        loss_accum = 0
        
        for batch in loop:
            batch = batch.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            
            loss_accum += loss.item()
            loop.set_postfix(loss=loss.item())
            
    return model

#inference
def inference(model):
    model.eval()
    results = []
    
    test_folders = sorted([
        os.path.join(CONFIG['TEST_DIR'], f) 
        for f in os.listdir(CONFIG['TEST_DIR']) 
        if os.path.isdir(os.path.join(CONFIG['TEST_DIR'], f))
    ])

    print("Starting inference")

    with torch.no_grad():
        for folder_path in tqdm(test_folders):
            video_id_raw = os.path.basename(folder_path)
            video_id_clean = str(int(video_id_raw))
            
            frames = sorted([
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg'))
            ])
            frames.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]))
            

            video_tensor = []
            for f in frames:
                try:
                    img = Image.open(f).convert('RGB')
                    img = common_transforms(img)
                    video_tensor.append(img)
                except:
                    pass
            
            
            full_video = torch.stack(video_tensor).to(CONFIG['DEVICE'])
            frame_scores = {i: [] for i in range(len(frames))}
            
            # Sliding Window
            seq_len = CONFIG['CLIP_LEN']
            
            for i in range(len(full_video) - seq_len + 1):
                clip = full_video[i : i+seq_len].permute(1, 0, 2, 3).unsqueeze(0)
                
                # Normal
                recon = model(clip)
                loss_normal = torch.mean((clip - recon)**2).item()
                
                # Flip
                clip_flipped = torch.flip(clip, dims=[4])
                recon_flipped = model(clip_flipped)
                loss_flipped = torch.mean((clip_flipped - recon_flipped)**2).item()
                
                score = min(loss_normal, loss_flipped)
                
                for j in range(i, i + seq_len):
                    frame_scores[j].append(score)
            
            for i, frame_path in enumerate(frames):
                frame_filename = os.path.basename(frame_path)
                frame_digits = re.findall(r'\d+', frame_filename)
                frame_num = int(frame_digits[-1]) if frame_digits else i + 1

                row_id = f"{video_id_clean}_{frame_num}"
                
                if frame_scores[i]:
                    avg_score = np.mean(frame_scores[i])
                else:
                    avg_score = 0.0
                
                results.append([row_id, avg_score])

    df = pd.DataFrame(results, columns=['Id', 'Predicted'])
    
    min_val, max_val = df['Predicted'].min(), df['Predicted'].max()
    if max_val > min_val:
        df['Predicted'] = (df['Predicted'] - min_val) / (max_val - min_val)
    else:
        df['Predicted'] = 0.0
        
    df.to_csv(CONFIG['SUBMISSION_FILE'], index=False)
    print("Done")


if __name__ == "__main__":
    trained_model = train()
    inference(trained_model)
