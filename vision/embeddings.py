#dino embeddings
dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dino_model.eval().to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def dino_embeds(path):
    embeds = []
    names = []
    for p in os.listdir(path):
        full_path = os.path.join(path, p)
        img = Image.open(full_path).convert("RGB")
        if img is None:
            continue  
        x = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = dino_model(x)          
        embeds.append(emb)
        names.append(p)
    if len(embeds) == 0:
        return None, None
    embeds = torch.cat(embeds).squeeze(-1).squeeze(-1)  
    embeds = F.normalize(embeds, p=2, dim=1)
    return embeds, names

# clip embeddings
import torch
import os
from PIL import Image
import torch.nn.functional as F
import clip 

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

def clip_embeds(path):
    embeds = []
    names = []

    for p in os.listdir(path):
        full_path = os.path.join(path, p)
        img = Image.open(full_path).convert("RGB")
        if img is None:
            continue
        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(x)

        embeds.append(emb)
        names.append(p)

    if len(embeds) == 0:
        return None, None

    embeds = torch.cat(embeds).squeeze(-1).squeeze(-1) 
    embeds = F.normalize(embeds, p=2, dim=1)  

  ## resnet embeddings
  from torchvision import transforms

class resnetEmb(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),                             
            transforms.Normalize(                              
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def forward(self, path):
        embeds = []
        names = []
        with torch.no_grad():                                    
            for p in os.listdir(path):
                img_path = os.path.join(path, p)
                img = crop_face(img_path) 
                if img is None:
                    continue
                x = self.transform(img).unsqueeze(0).to(device) 
                embed = self.model(x)
                embeds.append(embed)
                names.append(p)
                
        return embeds, names

    return embeds, names
