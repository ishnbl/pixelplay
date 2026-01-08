import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.transforms import CenterCrop
import numpy as np
import cv2
import glob, os, re
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis

#Hyeperparameters
config = {
    'train': 'dataset/training_videos', 
    'test': 'dataset/testing_videos',
    'batch': 16, 
    'clip': 16, 
    'size': 128,
    'crop': 112,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Load ResNet3d
def get_feature_extractor():
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    model.fc = nn.Identity() # Remove classifier
    return model.to(config['device']).eval()


class Dataset(Dataset):
    def __init__(self, root, mode='train'):
        self.mode = mode
        self.clips = []
        self.crop = CenterCrop(config['crop'])
        # Normalisation to normalize according to Kinetics-400 dataset
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3,1,1,1)
        self.std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3,1,1,1)
        
        stride = 4 if mode == 'train' else 1
        for folder in sorted(glob.glob(f"{root}/*")):
            frames = sorted(glob.glob(f"{folder}/*.jpg"), key=lambda x: int(re.search(r'\d+', x).group()))
            if len(frames) < config['clip']: continue
            
            vid_id = os.path.basename(folder)
            for i in range(0, len(frames) - config['clip'] + 1, stride):
                mid_frame = re.search(r'\d+', os.path.basename(frames[i + config['clip']//2])).group()
                self.clips.append((frames[i:i+config['clip']], f"{int(vid_id)}_{int(mid_frame)}"))

    def __len__(self): return len(self.clips)

    def align_sequence(self, frames):

        aligned_frames = [frames[0]]
        
        for i in range(1, len(frames)):
            prev_frame = aligned_frames[-1]
            curr_frame = frames[i]
            curr_flipped = cv2.flip(curr_frame, 0) 
            
            # MSE Calculation
            mse_normal = np.mean((prev_frame - curr_frame) ** 2)
            mse_flipped = np.mean((prev_frame - curr_flipped) ** 2)
            
            # Save the one with min error
            if mse_flipped < mse_normal:
                aligned_frames.append(curr_flipped)
            else:
                aligned_frames.append(curr_frame)
                
        return np.stack(aligned_frames)

    def __getitem__(self, idx):
        paths, clip_id = self.clips[idx]
        imgs = []
        
        # Load Raw Images
        for p in paths:
            img = cv2.imread(p)
            img = cv2.resize(img, (config['size'], config['size'])) if img is not None else np.zeros((config['size'], config['size'], 3))
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        # Apply Alignment Logic by flipping
        if self.mode == 'test':
            buffer_np = self.align_sequence(imgs)
        else:
            buffer_np = np.stack(imgs)
            
        # Preprocess before sending to model
        buffer = torch.from_numpy(buffer_np).float().permute(3, 0, 1, 2) / 255.0
        buffer = self.crop((buffer - self.mean) / self.std)
        
        return buffer, clip_id

# Feature Extraction
def get_embeddings(loader, model):
    feats, ids = [], []
    with torch.no_grad():
        for clips, batch_ids in tqdm(loader, desc="Extracting"):
            feats.append(model(clips.to(config['device'])).cpu().numpy())
            ids.extend(batch_ids)
    return np.concatenate(feats), ids

if __name__ == '__main__':
    model = get_feature_extractor()
    
    # Learn normal data distribution
    print("Learning normal data distribution")
    train_ds = Dataset(config['train'], mode='train')
    train_loader = DataLoader(train_ds, batch_size=config['batch'], shuffle=False, num_workers=4)
    
    train_feats, _ = get_embeddings(train_loader, model)
    
    # Fit Gaussian
    mean_vec = np.mean(train_feats, axis=0)
    cov_mat = np.cov(train_feats, rowvar=False) + np.eye(512) * 1e-4
    inv_cov = np.linalg.inv(cov_mat)
    
    # Test with flipping
    print("Testing with flipping")
    test_ds = Dataset(config['test'], mode='test')
    test_loader = DataLoader(test_ds, batch_size=config['batch'], shuffle=False, num_workers=4)
    
    final_scores = []
    final_ids = []

    with torch.no_grad():
        for clips, batch_ids in tqdm(test_loader, desc="Inference"):
            clips = clips.to(config['device']) 
            
            # Forward Pass on Aligned Clip
            feats_orig = model(clips).cpu().numpy()
            

            clips_flip = torch.flip(clips, dims=[3]) 
            feats_flip = model(clips_flip).cpu().numpy()
            
            # Calculate Scores
            for i in range(len(feats_orig)):
                #Calculate mahalanobis distance
                d_orig = mahalanobis(feats_orig[i], mean_vec, inv_cov)
                d_flip = mahalanobis(feats_flip[i], mean_vec, inv_cov)
                
                # Take Minimum
                final_scores.append(min(d_orig, d_flip))
            
            final_ids.extend(batch_ids)

    df = pd.DataFrame({'Id': final_ids, 'Predicted': final_scores})
    #regex search of images
    all_frames = []
    for f in glob.glob(f"{config['test']}/*/*.jpg"):
        vid = int(os.path.basename(os.path.dirname(f)))
        fr = int(re.search(r'\d+', os.path.basename(f)).group())
        all_frames.append(f"{vid}_{fr}")
        
    final = pd.DataFrame({'Id': all_frames}).merge(df.groupby('Id', as_index=False).max(), on='Id', how='left')
    final[['vid', 'fr']] = final['Id'].str.split('_', expand=True).astype(int)
    final = final.sort_values(['vid', 'fr'])
    
    # Fill gaps
    final['Predicted'] = final.groupby('vid')['Predicted'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    norm_func = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
    final['Predicted'] = final.groupby('vid')['Predicted'].transform(norm_func).fillna(0)
    
    final[['Id', 'Predicted']].to_csv('submission_mahal.csv', index=False)
    print("Done")
