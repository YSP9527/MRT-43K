import os
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description="Image Classification Training Script")
    
 
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    
  
    parser.add_argument("--train_csv", required=True, type=str, help="Path to training CSV file")
    parser.add_argument("--val_csv", required=True, type=str, help="Path to validation CSV file")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    
 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--freeze_features", action="store_true", help="Freeze ResNet backbone")
    

    parser.add_argument("--output_dir", type=str, default="./work_dirs", help="Root dir for logs and ckpts")
    parser.add_argument("--exp_name", type=str, default="exp1", help="Experiment name/round")
    
    return parser.parse_args()

class ImageDataset_from_csv(Dataset):
    def __init__(self, csv_file, num_classes, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.num_classes = num_classes
        
        if 'image_path' not in self.data.columns or 'label' not in self.data.columns:
            raise ValueError("CSV must contain 'image_path' and 'label' columns")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = int(self.data.iloc[idx]['label'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(num_classes, freeze_features=True, device='cuda'):

    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet50(pretrained=True)
    
    if freeze_features:
        for param in model.parameters(): 
            param.requires_grad = False
        print("=> Backbone frozen, training only the FC layer.")
    
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model.to(device)

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / total, correct / total

def save_checkpoint(path, model, optimizer, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def main():
    args = get_args()
    
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

   
    exp_root = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir = os.path.join(exp_root, "checkpoints")
    log_dir = os.path.join(exp_root, "runs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

 
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    train_dataset = ImageDataset_from_csv(args.train_csv, args.num_classes, transform=data_transform)
    val_dataset = ImageDataset_from_csv(args.val_csv, args.num_classes, transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

  
    model = create_model(args.num_classes, freeze_features=args.freeze_features, device=device)
    criterion = nn.CrossEntropyLoss()
    
    params_to_update = model.fc.parameters() if args.freeze_features else model.parameters()
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    
    writer = SummaryWriter(log_dir)
    best_val_acc = 0.0


    for epoch in range(args.epochs):
        model.train()
        total_loss, total_samples = 0.0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / total_samples
        avg_val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")

     
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

 
        if (epoch % 10 == 0) or (val_acc > best_val_acc):
            suffix = "best" if val_acc > best_val_acc else f"ep{epoch}"
            ckpt_name = f"resnet50_{args.exp_name}_{suffix}_acc{val_acc:.4f}.pth"
            save_checkpoint(os.path.join(ckpt_dir, ckpt_name), model, optimizer, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"🏆 New best model saved with accuracy: {val_acc:.4f}")

    writer.close()
    print(f"🏁 Training finished. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()