import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import random
import os
from training.lanenet.LaneDataset import LaneDataset
from models.lanenet.model import LaneNet

import matplotlib.pyplot as plt

# Centralized image shape (width, height)
IMAGE_SHAPE = (227, 128)  # (width, height)

def get_row_y_positions(img_height):
    # 10 rows, from 85% to 35.5%, each 5.5% apart (from bottom to top)
    return [int(img_height * (0.85 - i * 0.055)) for i in range(10)]

def draw_lanes_on_image(image, x_positions, visibility, color_left=(0,0,255), color_right=(0,255,0)):
    """
    Draws lane points on the image based on model output.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    y_positions = get_row_y_positions(height)

    # Left lane: lane_idx=0
    for i in range(10):
        vis = torch.sigmoid(visibility[0, i]).item()
        x = x_positions[0, i].item()
        if vis > 0.5:
            x_pixel = int(x * width)
            y_pixel = y_positions[i]
            draw.ellipse((x_pixel-4, y_pixel-4, x_pixel+4, y_pixel+4), fill=color_left)

    # Right lane: lane_idx=1
    for i in range(10):
        vis = torch.sigmoid(visibility[1, i]).item()
        x = x_positions[1, i].item()
        if vis > 0.5:
            x_pixel = int(x * width)
            y_pixel = y_positions[i]
            draw.ellipse((x_pixel-4, y_pixel-4, x_pixel+4, y_pixel+4), fill=color_right)

    return image

def main():
    # Settings
    dataset_dir = 'data/processed_tusimple'
    images_dir = os.path.join(dataset_dir, 'images')
    model_path = 'training/lanenet/model_files/lanenet_tusimple_best.pth'
    num_images = 4  # Number of images in the collage

    # Device selection: MPS > CUDA > CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Image transform (should match training)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SHAPE[1], IMAGE_SHAPE[0])),  # (height, width)
        transforms.ToTensor(),
    ])

    # Load model
    model = LaneNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get random images from the images directory
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    chosen_files = random.sample(image_files, min(num_images, len(image_files)))

    collage_images = []
    for fname in chosen_files:
        # Load and preprocess image
        orig_img = Image.open(os.path.join(images_dir, fname)).convert("RGB")
        img = transform(orig_img).unsqueeze(0).to(device)

        # Run model
        with torch.no_grad():
            x_positions, visibility_logits = model(img)
            # x_positions, visibility_logits: [1, 2, 10]
            x_positions = x_positions[0]  # [2, 10]
            visibility_logits = visibility_logits[0]  # [2, 10]

        # Draw lanes on a copy of the original image (resize to match transform)
        vis_img = orig_img.resize(IMAGE_SHAPE)  # (width, height)
        vis_img = draw_lanes_on_image(vis_img, x_positions, visibility_logits)
        collage_images.append(vis_img)

    # Create collage
    cols = 2
    rows = (len(collage_images) + 1) // 2
    collage_width = IMAGE_SHAPE[0] * cols
    collage_height = IMAGE_SHAPE[1] * rows
    collage = Image.new('RGB', (collage_width, collage_height), (0,0,0))

    for idx, img in enumerate(collage_images):
        x = (idx % cols) * IMAGE_SHAPE[0]
        y = (idx // cols) * IMAGE_SHAPE[1]
        collage.paste(img, (x, y))

    # Show and save
    collage.show()
    collage.save("lanenet_collage.png")
    print("Collage saved as lanenet_collage.png")

if __name__ == "__main__":
    main()
