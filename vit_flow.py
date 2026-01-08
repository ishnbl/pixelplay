import os
import glob
import re
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

#Hyperparameters
CONFIG = {
    'SEQ_LEN': 10,
    'SIZE': 128,
    'PATCH': 16,
    'EMBED': 64,
    'DEPTH': 4,
    'HEADS': 4,
    'BATCH': 16,
    'EPOCHS': 1,
    'LR': 1e-3,
    'ROOT': './dataset',
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}



class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.patch_embed = nn.Conv2d(CONFIG['SEQ_LEN'], CONFIG['EMBED'], 
                                     kernel_size=CONFIG['PATCH'], stride=CONFIG['PATCH'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=CONFIG['EMBED'], nhead=CONFIG['HEADS'], batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=CONFIG['DEPTH'])
        self.head = nn.Linear(CONFIG['EMBED'], CONFIG['PATCH']**2)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = self.encoder(x)
        x = self.head(x)
        
        b = x.shape[0]
        h = w = CONFIG['SIZE'] // CONFIG['PATCH']
        p = CONFIG['PATCH']
        x = x.view(b, h, w, p, p)
        x = torch.einsum('nhwpq->nhpwq', x)
        return x.reshape(b, 1, CONFIG['SIZE'], CONFIG['SIZE'])


class Dataset(Dataset):
    def __init__(self, root, is_train):
        self.is_train = is_train
        self.samples = []
        self.my_list = {} 

        folder = 'training_vol' if is_train else 'testing_vol'
        path = os.path.join(root, folder)
        files = sorted(glob.glob(os.path.join(path, '*.mat')))


        for f in tqdm(files):
            # grab ids
            vid_id = int(re.findall(r'\d+', os.path.basename(f))[0])
            
            mat = scipy.io.loadmat(f)
            data = mat['vol'].transpose(2, 0, 1) / 255.0
            

            tensor = torch.FloatTensor(data)
            tensor = tensor.unsqueeze(1)
            tensor = torch.nn.functional.interpolate(tensor, size=(CONFIG['SIZE'], CONFIG['SIZE']))
            self.my_list[vid_id] = tensor.squeeze(1)
            
            num_frames = self.my_list[vid_id].shape[0]
            
            if is_train:
                for i in range(CONFIG['SEQ_LEN'], num_frames):
                    self.samples.append((vid_id, i))
            else:
                for i in range(num_frames):
                    self.samples.append((vid_id, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid_id, frame_idx = self.samples[idx]
        
        video_data = self.my_list[vid_id]
        
        target = video_data[frame_idx].unsqueeze(0)
        
        # Get the history to input
        if frame_idx >= CONFIG['SEQ_LEN']:
            inp = video_data[frame_idx-CONFIG['SEQ_LEN'] : frame_idx]
        else:
            # if len of histroy not enough then repeat first frame
            pad = video_data[0].unsqueeze(0).repeat(CONFIG['SEQ_LEN'], 1, 1)
            inp = pad
            
        return inp, target, vid_id, frame_idx + 1

#Train
def train():
    ds = Dataset(CONFIG['ROOT'], is_train=True)
    loader = DataLoader(ds, batch_size=CONFIG['BATCH'], shuffle=True)
    model = ViT().to(CONFIG['DEVICE'])
    opt = optim.AdamW(model.parameters(), lr=CONFIG['LR'])
    loss_fn = nn.MSELoss()
    
    print("Training")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        loop = tqdm(loader)
        for x, y, _, _ in loop:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item())
    return model

def predict(model):
    ds = Dataset(CONFIG['ROOT'], is_train=False)
    loader = DataLoader(ds, batch_size=CONFIG['BATCH'], shuffle=False)
    model.eval()
    results = []
    loss_fn = nn.MSELoss(reduction='none')
    
    print("Inference")
    with torch.no_grad():
        for x, y, vid, fnum in tqdm(loader):
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            
            #Normal
            p1 = model(x)
            err1 = loss_fn(p1, y).mean(dim=[1,2,3])
            
            # Flipped
            x_f = torch.flip(x, [2])
            y_f = torch.flip(y, [2])
            p2 = model(x_f)
            err2 = loss_fn(p2, y_f).mean(dim=[1,2,3])
            #Min error
            final_err = torch.min(err1, err2).cpu().numpy()
            
            for i in range(len(final_err)):
                row_id = f"{vid[i].item()}_{fnum[i].item()}"
                results.append([row_id, final_err[i]])

    df = pd.DataFrame(results, columns=['Id', 'Raw'])
    # Normalize
    df['Predicted'] = (df['Raw'] - df['Raw'].min()) / (df['Raw'].max() - df['Raw'].min())
    df[['Id', 'Predicted']].to_csv('submission_vit.csv', index=False)
    print("done")

if __name__ == "__main__":
    model = train()
    predict(model)
