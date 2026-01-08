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
from torchvision import transforms
from tqdm import tqdm

#HYPERPARAMETERS
CONFIG = {
    'BATCH_SIZE': 32,
    'EPOCHS': 5,
    'LR': 1e-3,
    'IMG_SIZE': 128,
    'PATCH_SIZE': 16,
    'EMBED_DIM': 384,
    'ENC_DEPTH': 12,
    'PRED_DEPTH': 6,
    'HEADS': 6,
    'MASK_RATIO': 0.4,
    'EMA_DECAY': 0.996,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

class VOL_DATA(Dataset):
    def __init__(self, root, split):
        self.files = sorted(glob.glob(os.path.join(root, split, '*.mat')))
        self.samples = []
        self.cache = {}
        self.transform = transforms.Resize((CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']))

        print("Loading")
        for fpath in tqdm(self.files):

            # Uses regex to get digits from filename
            match = re.search(r'\d+', os.path.basename(fpath))
            if not match: continue
            vid_id = match.group()
            mat = scipy.io.loadmat(fpath)
            # handle different mat keys automatically
            key = next(k for k in mat.keys() if k == 'vol')
            data = mat[key]
            
            # COnverts data: H, W, T to T, 1, H, W
            data = data.transpose(2, 0, 1)
            
            # Scale data to 0 to 1
            data = (data - data.min()) / (data.max() - data.min() + 1e-6)
            data = torch.tensor(data[:, np.newaxis, :, :], dtype=torch.float32)
            
            self.cache[vid_id] = data
            for i in range(len(data)):
                self.samples.append((vid_id, i))

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frame = self.samples[idx]
        img = self.transform(self.cache[vid][frame])
        return img, vid, frame + 1

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, depth):
        super().__init__()
        dim, patch = CONFIG['EMBED_DIM'], CONFIG['PATCH_SIZE']
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        num_patches = (CONFIG['IMG_SIZE'] // patch) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, CONFIG['HEADS']) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # (B, N, Dim)
        x = x + self.pos_embed
        
        if mask is not None: 
            x = x * (1 - mask.unsqueeze(-1))
            
        for block in self.blocks: x = block(x)
        return self.norm(x)

#JEPA MODEL
class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        dim = CONFIG['EMBED_DIM']
        
        # Components
        self.context_encoder = ViTEncoder(depth=CONFIG['ENC_DEPTH'])
        self.target_encoder = ViTEncoder(depth=CONFIG['ENC_DEPTH']) # The Teacher
        self.predictor = nn.Sequential(
            *[TransformerBlock(dim, CONFIG['HEADS']) for _ in range(CONFIG['PRED_DEPTH'])],
            nn.LayerNorm(dim)
        )
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        # Initialise Target Encoder as clone of Context Encoder
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters(): p.requires_grad = False

    #Update the weights of target encoder using EMA
    def update_target_ema(self):
        m = CONFIG['EMA_DECAY']
        with torch.no_grad():
            for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
                p_tgt.data.mul_(m).add_(p_ctx.data, alpha=1 - m)

    def forward(self, x, mask):
        # 1. Context Encoder (Sees masked image)
        ctx_repr = self.context_encoder(x, mask)
        
        # 2. Predictor (Tries to fill gaps)
        # Replace masked tokens with learnable [MASK] token
        B, N, D = ctx_repr.shape
        mask_bool = mask.unsqueeze(-1).expand(-1, -1, D).bool()
        pred_input = torch.where(mask_bool, self.mask_token.expand(B, N, -1), ctx_repr)
        
        pred_repr = self.predictor(pred_input) # Prediction
        
        # 3. Target Encoder the one using EMA seeing full image
        with torch.no_grad():
            target_repr = self.target_encoder(x, mask=None)
            
        return pred_repr, target_repr

#Randomly masks patches
def generate_mask(B, N):
    mask = torch.zeros(B, N)
    grid = int(np.sqrt(N))
    for b in range(B):
        for _ in range(np.random.randint(4, 8)):
            max_h = max(2, grid // 2)
            h = np.random.randint(2, max_h + 1)
            w = np.random.randint(2, max_h + 1)
            
            y = np.random.randint(0, max(1, grid - h))
            x = np.random.randint(0, max(1, grid - w))
            
            for i in range(y, y+h):
                for j in range(x, x+w):
                    idx = i*grid + j
                    if idx < N:
                        mask[b, idx] = 1
    return mask.to(CONFIG['DEVICE'])

def train_and_infer():
    model = JEPA().to(CONFIG['DEVICE'])
    opt = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    


    train_ds = VOL_DATA(DATA_ROOT, 'training_vol')
    train_loader = DataLoader(train_ds, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
    
    # B. Training Loop
    print("Training JEPA Model")
    embeddings = []
    
    #Calculate number of patches
    num_patches = (CONFIG['IMG_SIZE'] // CONFIG['PATCH_SIZE']) ** 2
    
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for img, _, _ in loop:
            img = img.to(CONFIG['DEVICE'])
            
            #generate masks
            mask = generate_mask(img.size(0), num_patches)
            
            # Forward
            pred, target = model(img, mask)
            
            # Loss: MSE only on masked regions
            loss = F.mse_loss(pred[mask.bool()], target[mask.bool()])
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target_ema()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    #Build Normal Distribution for mahalanobis distance calculation, i also used with a combination of direct l2 loss on embeddings too which gave slightly better results
    print("Building Normal Distribution")
    model.eval()
    with torch.no_grad():
        for img, _, _ in tqdm(train_loader):
            # Extract features Encoder
            feat = model.target_encoder(img.to(CONFIG['DEVICE'])).mean(dim=1).cpu().numpy()
            embeddings.append(feat)
            
    embeddings = np.vstack(embeddings)
    mean_vec = np.mean(embeddings, axis=0)
    cov_mat = np.cov(embeddings, rowvar=False) + np.eye(embeddings.shape[1]) * 1e-6

    # Inference
    print("Generating Submission")
    test_ds = VOL_DATA(DATA_ROOT, 'testing_vol')
    test_loader = DataLoader(test_ds, batch_size=1)
    inv_cov = np.linalg.inv(cov_mat)
    results = []

    with torch.no_grad():
        for img, vid, frame in tqdm(test_loader):
            img = img.to(CONFIG['DEVICE'])
            
            # Anomaly Score = Mahalanobis Distance
            feat = model.target_encoder(img).mean(dim=1).cpu().numpy()
            diff = feat - mean_vec
            score = np.sqrt(np.sum((diff @ inv_cov) * diff, axis=1))[0]
            
            # Clean IDs
            vid_clean = re.sub(r'\D', '', str(vid[0]))
            
            results.append({'Id': f"{int(vid_clean)}_{frame.item()}", 'score': score})

    # Save
    df = pd.DataFrame(results)
    #Do per video norm
    df['vid_key'] = df['Id'].apply(lambda x: str(x).split('_')[0])

    def group_min_max(x):
        if x.max() == x.min(): return x - x
        return (x - x.min()) / (x.max() - x.min())

    df['Predicted'] = df.groupby('vid_key')['score'].transform(group_min_max)
        
    df[['Id', 'Predicted']].to_csv('submission_flow_jepa.csv', index=False)
    print("Done")


if __name__ == "__main__":
    DATA_ROOT = './dataset' 
    train_and_infer()
