import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device=device)

def text_embedding(sen):
    text = clip.tokenize(sen).to(device)
    text_features = model.encode_text(text)
    return text_features