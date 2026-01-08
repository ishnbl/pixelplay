import os
import glob
import numpy as np
import pandas as pd
import cv2
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re  

#Hyeperparameters

CONFIG = {
    'SEQ_LEN': 4,              
    'IMG_SIZE': 128,          
    'BATCH_SIZE': 16,
    'EPOCHS': 1,
    'LR': 1e-4,
    'ROOT_DIR': './dataset',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}


#LSTM Cell
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, input_tensor, state):
        h, c = state
        combined = torch.cat([input_tensor, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        
        next_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        next_h = torch.sigmoid(o) * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, height, width, device):
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=device))

class ConvLSTMPredictor(nn.Module):
    def __init__(self):
        super(ConvLSTMPredictor, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        
        # ConvLSTM
        self.lstm = ConvLSTMCell(in_channels=32, hidden_channels=64, kernel_size=3, bias=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        b, seq, c, h, w = x.size()
        
        hidden = self.lstm.init_hidden(b, h//4, w//4, x.device)
        
        # Loop through the sequence
        for t in range(seq):
            features = self.encoder(x[:, t])
            hidden = self.lstm(features, hidden)
        
        # Predict the next frame using the final hidden state
        return self.decoder(hidden[0])

#Dataset
class Dataset(Dataset):
    def __init__(self, root_dir, is_train, apply_flip=False):
        self.seq_len = CONFIG['SEQ_LEN']
        self.is_train = is_train
        self.apply_flip = apply_flip
        self.samples = []
        self.cache = {}

        folder = 'training_videos' if is_train else 'testing_videos'
        self.load_images(os.path.join(root_dir, folder))


    def load_images(self, path):
        if not os.path.exists(path): return
        video_folders = sorted(os.listdir(path))
        
        for vid in video_folders:
            vid_path = os.path.join(path, vid)
            if not os.path.isdir(vid_path): continue
            
            all_files = os.listdir(vid_path)
            files = sorted([f for f in all_files if f.lower().endswith(('.jpg'))])
            
            full_paths = [os.path.join(vid_path, f) for f in files]
            
            if len(files) < self.seq_len + 1:
                continue

            # Create sliding windows
            for i in range(len(full_paths) - self.seq_len):
                #get ids
                target_filename = files[i+self.seq_len]
                match = re.search(r'\d+', target_filename)
                
                frame_num = int(match.group())


                self.samples.append({
                    'vid': vid,
                    'frame_id': frame_num,
                    'input': full_paths[i : i+self.seq_len],
                    'target': full_paths[i+self.seq_len]
                })


    def process_data(self, data):
        img = cv2.imread(data, 0) # Load as Grayscale
        
        
        img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE'])) / 255.0

            
        # Vertical Flip
        if self.apply_flip:
            img = cv2.flip(img, 0) 
            
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        inputs = torch.stack([self.process_data(p) for p in item['input']])
        target = self.process_data(item['target'])


        if self.is_train:
            return inputs, target
        else:
            return inputs, target, str(item['vid']), str(item['frame_id'])

#train
def train_model():
    dataset = Dataset(CONFIG['ROOT_DIR'], is_train=True)


    loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
    
    model = ConvLSTMPredictor().to(CONFIG['DEVICE'])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.MSELoss()
    
    print("Training")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for inputs, target in loop:
            inputs, target = inputs.to(CONFIG['DEVICE']), target.to(CONFIG['DEVICE'])
            
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
            
    return model

#inference
def inference(model):
    if model is None: return

    model.eval()
    results = {}
    
    print("Inference")
    
   
    ds = Dataset(CONFIG['ROOT_DIR'], is_train=False, apply_flip=False)
    
    loader = DataLoader(ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for inputs, target, vids, frames in tqdm(loader):
            inputs, target = inputs.to(CONFIG['DEVICE']), target.to(CONFIG['DEVICE'])
            
            prediction = model(inputs)
            
            # Calculate error without flipping
            mse_normal = torch.mean((prediction - target)**2, dim=[1, 2, 3])
            
            #Calculate error with flipping
            target_flipped = torch.flip(target, [2])
            mse_flipped = torch.mean((prediction - target_flipped)**2, dim=[1, 2, 3])
            
            # Take the min error
            best_mse = torch.min(mse_normal, mse_flipped).cpu().numpy()
            
            for i in range(len(best_mse)):
                vid_clean = re.sub(r'\D', '', vids[i]) 
                frame_clean = frames[i]
                
                key = f"{int(vid_clean)}_{int(frame_clean)}"
                
                if key not in results: results[key] = []
                results[key].append(best_mse[i])

    final_rows = []
    
    sorted_keys = sorted(results.keys(), key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1])))
    
    for key in sorted_keys:
        best_score = min(results[key])
        final_rows.append([key, best_score])
        
    df = pd.DataFrame(final_rows, columns=['Id', 'Raw'])
    
    df['Predicted'] = (df['Raw'] - df['Raw'].min()) / (df['Raw'].max() - df['Raw'].min())
    filename = f"submission_img_flip.csv"
    df[['Id', 'Predicted']].to_csv(filename, index=False)
    print("saved")


if __name__ == "__main__":
    trained_model = train_model()
    inference(trained_model)
