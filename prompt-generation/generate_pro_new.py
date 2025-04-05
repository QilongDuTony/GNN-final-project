import torch
from torchvision import transforms
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from encoder_decoder import EncoderCNN, ImageCaptioningModel

# Config
embed_dim = 512
image_path = "./distant_desert.jpeg"  # ./data/diffusiondb_images/image_0000.png
model_path = "./best_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and decoder
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
decoder = GPT2LMHeadModel.from_pretrained("gpt2")

encoder = EncoderCNN(embed_dim).to(device)
model = ImageCaptioningModel(encoder, decoder, embed_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

image = Image.open(image_path).convert("RGB")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    image_embedding = model.encoder(image_tensor)
    image_embedding = model.fc(image_embedding).unsqueeze(1)

print("Initial image_embedding shape:", image_embedding.shape)

# Generate prompt
generated_ids = []
input_embeds = image_embedding
max_length = 50

for step in range(max_length):
    print(f"\nStep {step + 1}:")
    print("input_embeds shape:", input_embeds.shape)

    outputs = model.decoder(inputs_embeds=input_embeds)
    logits = outputs.logits
    print("logits shape:", logits.shape)

    if logits.dim() == 2:
        logits = logits.unsqueeze(0)

    next_token_logits = logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated_ids.append(next_token_id.item())

    print("next_token_id:", next_token_id.item())

    if next_token_id.item() == tokenizer.eos_token_id:
        break

    # Convert token_id to embedding: [1, 1, hidden_size]
    token_embed = model.decoder.transformer.wte(next_token_id)
    token_embed = token_embed.unsqueeze(1)
    print("token_embed shape:", token_embed.shape)

    if input_embeds.dim() == 4:
        input_embeds = input_embeds.squeeze(1)
        print("Squeezed input_embeds to shape:", input_embeds.shape)

    input_embeds = torch.cat((input_embeds, token_embed), dim=1)
    print("Updated input_embeds shape:", input_embeds.shape)

prompt = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("\nGenerated Prompt:")
print(prompt)
