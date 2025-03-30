import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        # Freeze ResNet weights
        with torch.no_grad():
            features = self.resnet(images).squeeze(-1).squeeze(-1)
        embeddings = self.fc(features)
        return embeddings


class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, embed_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Match encoder output to GPT2 hidden size
        self.fc = nn.Linear(embed_dim, decoder.config.n_embd)

    def forward(self, images, captions=None, attention_mask=None):
        img_embed = self.encoder(images)
        img_embed = self.fc(img_embed).unsqueeze(1)

        if captions is not None:
            inputs_embeds = self.decoder.transformer.wte(captions)
            inputs_embeds = torch.cat((img_embed, inputs_embeds), dim=1)

            # Adjust attention mask to account for image token
            if attention_mask is not None:
                prefix_mask = torch.ones(
                    (attention_mask.size(0), 1), dtype=attention_mask.dtype
                ).to(attention_mask.device)
                attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)

            # Pad labels to match inputs_embeds length
            if captions.size(1) < inputs_embeds.size(1):
                pad_token_id = (
                    self.decoder.config.pad_token_id or tokenizer.eos_token_id
                )
                padding = torch.full(
                    (captions.size(0), 1),
                    pad_token_id,
                    dtype=captions.dtype,
                    device=captions.device,
                )
                captions = torch.cat((padding, captions), dim=1)

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=captions,
            )
            return outputs
        else:
            return img_embed


def tokenizer_and_decoder():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    decoder = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, decoder


class PromptDataset(Dataset):
    def __init__(self, csv_file, image_dir, tokenizer, transform=None, max_length=64):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row["file_name"])
        prompt = row["prompt"]

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return image, input_ids, attention_mask


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    batch_losses = []

    for images, input_ids, attention_mask in dataloader:
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(images, captions=input_ids, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        batch_losses.append(loss.item())

    return total_loss / len(dataloader), batch_losses


if __name__ == "__main__":
    embed_dim = 512
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Updated paths for your full dataset
    csv_file = "./data/diffusiondb_sample.csv"
    image_dir = "./data/diffusiondb_images"

    tokenizer, decoder = tokenizer_and_decoder()
    encoder = EncoderCNN(embed_dim).to(device)
    model = ImageCaptioningModel(encoder, decoder, embed_dim).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = PromptDataset(csv_file, image_dir, tokenizer, transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    num_epochs = 20
    all_epoch_losses = []
    all_batch_losses = []
    for epoch in range(num_epochs):
        epoch_loss, batch_losses = train(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        all_epoch_losses.append(epoch_loss)
        all_batch_losses.extend(batch_losses)

    torch.save(model.state_dict(), "best_model.pth")
    print("Model saved as best_model.pth")
