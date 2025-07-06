import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


imsize = 256  # Fast execution size
transform = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image.to(device)

content_img = load_image("/content/WhatsApp Image 2024-12-03 at 19.43.24.jpeg")  # replace with your path
style_img = load_image("/content/__results___14_1.jpg")      # replace with your path
generated = content_img.clone().requires_grad_(True)

class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:29].to(device).eval()
        self.chosen = ["0", "5", "10", "19", "28"]

    def forward(self, x):
        features = []
        for name, layer in enumerate(self.model):
            x = layer(x)
            if str(name) in self.chosen:
                features.append(x)
        return features

model = VGGFeatures().to(device)

optimizer = optim.Adam([generated], lr=0.01)
total_steps = 100  # fast training
alpha = 1
beta = 0.01

for step in range(total_steps):
    gen_feats = model(generated)
    content_feats = model(content_img)
    style_feats = model(style_img)

    content_loss = style_loss = 0

    for gf, cf, sf in zip(gen_feats, content_feats, style_feats):
        content_loss += torch.mean((gf - cf)**2)

        # Gram matrix for style
        G = gf.view(gf.shape[1], -1)
        A = sf.view(sf.shape[1], -1)
        gram_g = G @ G.t()
        gram_a = A @ A.t()
        style_loss += torch.mean((gram_g - gram_a)**2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step [{step}/{total_steps}], Loss: {total_loss.item():.4f}")

save_image(generated, "stylized_output.png")

# Display
img = generated.clone().squeeze().detach().cpu().clamp(0, 1)
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
plt.title("Stylized Output")
plt.show()
