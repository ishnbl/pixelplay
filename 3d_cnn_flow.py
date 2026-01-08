import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#Hyperparams
CONFIG = {
    'BATCH_SIZE': 8,         
    'EPOCHS': 5,
    'LR': 1e-4,
    'SEQ_LEN': 16,            
    'IMG_SIZE': (128, 128),
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'DATA_ROOT': './dataset',
    'NUM_WORKERS': 2
}



class Dataset(Dataset):
    def __init__(self, root_dir, split='training_vol', seq_len=16):
        self.root_dir = os.path.join(root_dir, split)
        self.seq_len = seq_len
        self.samples = [] 
        self.data_cache = {} 
        # Find .mat files
        files = sorted(glob.glob(os.path.join(self.root_dir, '*.mat')))
        for fpath in tqdm(files):
            # Extract video id
            filename = os.path.basename(fpath)
            match = re.search(r'\d+', filename)
            if not match: continue
            vid_id = match.group()
            mat = scipy.io.loadmat(fpath)
            # Raw Shape: (H, W, Frames) -> e.g. (120, 160, 1364)
            data = mat['vol'] 
            # 1. Transpose to take the time dim back
            data = data.transpose(2, 0, 1)
            # 2. Normalize 0-1 
            data = (data / 255.0).astype(np.float32)
            self.data_cache[vid_id] = data
            num_frames = data.shape[0]
            for i in range(num_frames - seq_len):
                # Store tuple: (vid id, start frame)
                self.samples.append((vid_id, i))
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, start_idx = self.samples[idx]
        
        full_seq_numpy = self.data_cache[vid_id][start_idx : start_idx + self.seq_len + 1]
        
        # Convert to Tensor
        tensor = torch.from_numpy(full_seq_numpy).float()
        
        if tensor.shape[1:] != CONFIG['IMG_SIZE']:
            tensor = tensor.unsqueeze(1) 
            tensor = F.interpolate(tensor, size=CONFIG['IMG_SIZE'], mode='bilinear', align_corners=False)
            tensor = tensor.squeeze(1)   # Back to (17, H, W)
            
        input_seq = tensor[:-1]   
        target_frame = tensor[-1] 
        
        # Reshape for 3D Conv
        input_seq = input_seq.unsqueeze(0) 
        target_frame = target_frame.unsqueeze(0)
            
        return input_seq, target_frame, vid_id, start_idx + self.seq_len + 1


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        
        # ENCODER 3d
        self.encoder = nn.Sequential(
            # Input: (B, 1, 16, 128, 128)
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # (32, 8, 64, 64)
            
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), #(64, 4, 32, 32)
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # (128, 2, 16, 16)
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))  # (256, 1, 8, 8)
        )
        
        # DECODER 2d
        self.decoder = nn.Sequential(
            # Input: (B, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), #(128, 16, 16)
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), #(64, 32, 32)
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # (32, 64, 64)
            
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() #(1, 128, 128)
        )

    def forward(self, x):
        # 1. Encode Sequence
        x = self.encoder(x)
        x = x.squeeze(2)    
        # 3. Decode Frame
        x = self.decoder(x) 
        return x

#training
def train_model():
    print("training ")
    
    # Load Dataset
    train_dataset = Dataset(CONFIG['DATA_ROOT'], split='training_vol', seq_len=CONFIG['SEQ_LEN'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=CONFIG['NUM_WORKERS'])
    
    # Initialize Model
    model = C3D().to(CONFIG['DEVICE'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    
    # Training Loop
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        for input_seq, target_frame, _, _ in loop:
            input_seq = input_seq.to(CONFIG['DEVICE'])
            target_frame = target_frame.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            prediction = model(input_seq)
            loss = criterion(prediction, target_frame)
            loss.backward()
            optimizer.step() 
            train_loss += loss.item() * input_seq.size(0)
            loop.set_postfix(loss=loss.item())
            
    return model
#inference
def generate_submission(model):
    if model is None: return
    print("running inference")
    test_dataset = Dataset(CONFIG['DATA_ROOT'], split='testing_vol', seq_len=CONFIG['SEQ_LEN'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model.eval()
    criterion = nn.MSELoss(reduction='none') 
    results = []
    with torch.no_grad():
        for input_seq, target_frame, vid, frame in tqdm(test_loader, desc="Testing"):
            input_seq = input_seq.to(CONFIG['DEVICE'])
            target_frame = target_frame.to(CONFIG['DEVICE'])
            prediction = model(input_seq)
            
            # Anomaly Score = Reconstruction Error
            loss_map = criterion(prediction, target_frame)
            score = torch.mean(loss_map).item()
            id_str = f"{str(vid[0])}_{str(frame.item())}"
            results.append({'Id': id_str, 'Predicted': score})
            
    df = pd.DataFrame(results)
    # Min-Max Normalization to 0-1 range
    min_s = df['Predicted'].min()
    max_s = df['Predicted'].max()
    if max_s - min_s > 0:
        df['Predicted'] = (df['Predicted'] - min_s) / (max_s - min_s)
    else:
        df['Predicted'] = 0.0

    df.to_csv("submission_3d_vol2.csv", index=False)

if __name__ == "__main__":
    trained_model = train_model()
    generate_submission(trained_model)
