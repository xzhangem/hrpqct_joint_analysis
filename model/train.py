import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, global_max_pool, radius_graph
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sklearn.metrics as skm

# ====================== Dataset（保持你原来的，略微优化） ======================
class VariablePointCloudDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir) / split
        self.files = []
        self.labels = []
        
        for label, subfolder in enumerate(['healthy', 'diseased']):
            folder = self.root_dir / subfolder
            if folder.exists():
                for f in sorted(folder.glob('*.npy')):
                    self.files.append(f)
                    self.labels.append(label)
        
        print(f"[{split}] Loaded {len(self.files)} samples")

    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        pc = np.load(self.files[idx]).astype(np.float32)  # [N, 3]
        label = self.labels[idx]
        
        # 数据增强（你的原始增强保留）
        theta = np.random.uniform(0, 2 * np.pi)
        rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta),  np.cos(theta), 0],
                        [0, 0, 1]])
        pc = pc @ rot.T
        pc += np.random.normal(0, 0.02, pc.shape).astype(np.float32)
        pc = torch.from_numpy(pc).to(dtype=torch.float32)

        
        return Data(pos=pc, y=torch.tensor(label, dtype=torch.long))

def collate_fn(batch): return Batch.from_data_list(batch)

# ====================== 模型（你原来的，保持不变） ======================
class VariablePointNetPlusPlusBinary(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.sa1 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(3 + 3, 64), nn.BatchNorm1d(64), nn.ReLU(),   # ← 3 (rel pos) + 3 (input feat)
                nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU()
            ),
            global_nn=nn.Sequential(
                nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU()
            )
        )

        # Second SA layer: now input features are 256 → local_nn gets [3 + 256]
        self.sa2 = PointNetConv(
            local_nn=nn.Sequential(
                nn.Linear(3 + 256, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU()
            ),
            global_nn=nn.Sequential(
                nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU()
            )
        )

        '''
        self.sa1 = PointNetConv(
            local_nn=nn.Sequential(nn.Linear(in_channels+3,64),nn.BatchNorm1d(64),nn.ReLU(),
                                   nn.Linear(64,128),nn.BatchNorm1d(128),nn.ReLU()),
            global_nn=nn.Sequential(nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU())
        )
        self.sa2 = PointNetConv(
            local_nn=nn.Sequential(nn.Linear(256+3,256),nn.BatchNorm1d(256),nn.ReLU(),
                                   nn.Linear(256,512),nn.BatchNorm1d(512),nn.ReLU()),
            global_nn=nn.Sequential(nn.Linear(512,1024),nn.BatchNorm1d(1024),nn.ReLU())
        )
        '''

        self.fc1 = nn.Linear(1024,512)
        #self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,256)
        #self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 1)
        self.bn1 = nn.LayerNorm(512)
        self.bn2 = nn.LayerNorm(256)

    def forward(self, data):


        pos, batch = data.pos, data.batch

        # First layer: no input features yet → pass pos as both x and pos
        edge_index1 = radius_graph(pos, r=0.5, batch=batch, max_num_neighbors=64, loop=False)
        x1 = self.sa1(pos, pos, edge_index=edge_index1)           # ← correct & clean

        # Second layer: now we have features x1
        edge_index2 = radius_graph(pos, r=1.0, batch=batch, max_num_neighbors=128, loop=False)
        x2 = self.sa2(x1, pos, edge_index=edge_index2)            # ← pass (x, pos)

        # Global pooling
        x = global_max_pool(x2, batch)

        # Classification head
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        logit = self.fc3(x)

        return logit
    '''
        pos, batch = data.pos, data.batch
        edge_index1 = radius_graph(pos, r=0.5, batch=batch, max_num_neighbors=64)
        x1 = self.sa1((None, pos), (pos, pos), edge_index1)
        
        edge_index2 = radius(pos, pos, r=1.0, batch=batch, max_num_neighbors=128)
        x2 = self.sa2((x1, pos), (pos, pos), edge_index2)
        
        x = global_max_pool(x2, batch)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        return self.fc3(x)   # [B, 1] logit
    '''

# ====================== 训练主函数（推荐直接运行） ======================
def train_model(data_root="your_data_root", batch_size=2, epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载完整数据集并拆分
    full_dataset = VariablePointCloudDataset(data_root, split='train')
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)
    
    model = VariablePointNetPlusPlusBinary().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = batch.to(device=device)
            optimizer.zero_grad()
            logit = model(batch)
            loss = criterion(logit, batch.y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # ====================== 验证 ======================
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logit = model(batch)
                loss = criterion(logit, batch.y.float().unsqueeze(1))
                val_loss += loss.item()
                
                pred = torch.sigmoid(logit).cpu().numpy() > 0.5
                all_preds.extend(pred.flatten())
                all_labels.extend(batch.y.cpu().numpy())
        
        acc = skm.accuracy_score(all_labels, all_preds)
        auc = skm.roc_auc_score(all_labels, all_preds)
        
        scheduler.step()
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "pointnet2_best.pth")
            print(f"★ 新最佳模型已保存！Acc = {acc:.4f}")
        
        # 每 10 epoch 额外保存一次
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"pointnet2_epoch{epoch+1}.pth")
    
    print("训练完成！最佳模型已保存为 pointnet2_best.pth")
    return model

if __name__ == "__main__":
    train_model(data_root='./data/')   # ← 只改这一行路径即可
