import os
import argparse
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset, DataLoader

import utils
import vision_transformer as vits
from Calculation_function import clip_very_high_values, average_division_x_entropy

def get_args():
    parser = argparse.ArgumentParser(description="Feature extraction and entropy calculation script")
    
    parser.add_argument("--arch", default="vit_small", type=str, help="Model architecture")
    parser.add_argument("--patch_size", default=8, type=int, help="Patch size for ViT")
    parser.add_argument("--pretrained_weights", default="", type=str, help="Path to pretrained weights")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help="Key to load weights from")
    
    parser.add_argument("--image_size", default=224, type=int, help="Input image size")
    parser.add_argument("--input_csv", required=True, type=str, help="Path to the input dataset CSV")
    parser.add_argument("--img_root", required=True, type=str, help="Root directory of images")
    parser.add_argument("--output_dir", default="./output", type=str, help="Directory to save results")
    

    parser.add_argument("--split_num", default=5, type=int, help="Split number for entropy calculation")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of data loading workers")
    
    return parser.parse_args()

class GenericDataset(Dataset):
    def __init__(self, csv_file, img_root, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_root = img_root
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
    
        img_name = self.data.iloc[idx, 0]  
        img_path = os.path.join(self.img_root, img_name)
        label = self.data.iloc[idx, 1]     
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label, img_path

def main():
    args = get_args()
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_featmap = args.image_size // args.patch_size
    
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    
    if os.path.exists(args.pretrained_weights):
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    else:
        print(f"Warning: Pretrained weights not found at {args.pretrained_weights}. Training from scratch or using default init.")

    model.to(device)
    model.eval()


    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.image_size, args.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
    ])

    dataset = GenericDataset(args.input_csv, args.img_root, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True 
    )

 
    all_image_paths, all_labels, all_values = [], [], []

    for images, labels, image_paths in tqdm(dataloader, desc="Processing batches"):
        images = images.to(device, non_blocking=True)
        
        with torch.no_grad(): 
          
            attn = model.get_last_selfattention(images)
            attn = torch.mean(attn, dim=1)[:, 0, 1:]  
            attn = attn / (torch.sum(attn, dim=1, keepdim=True) + 1e-8)
        
        for i in range(images.size(0)):
            attn_np = attn[i].cpu().numpy().reshape(w_featmap, w_featmap)
            attn_np = clip_very_high_values(attn_np, 99.6)  
            avg_x_entropy = average_division_x_entropy(x=args.split_num, attn_np=attn_np) 

            all_values.append(avg_x_entropy)
            all_labels.append(labels[i].item())
            all_image_paths.append(image_paths[i])

  
    result_df = pd.DataFrame({
        'image_path': all_image_paths,
        'label': all_labels,
        'value': all_values
    })

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "results_with_value.csv")
    result_df.to_csv(output_path, index=False)

    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()