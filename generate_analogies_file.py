import random
import numpy as np
import yaml
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
from pathlib import Path


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


# Load the YAML file (modify the path as needed)
yaml_data = load_yaml('.\dataset\coco-2017\error_imgs.yaml')

# Get a list of all image identifiers (keys of the YAML data)
image_identifiers = list(yaml_data.keys())

# Randomly create groups of images
def create_random_groups(image_identifiers, group_size=6, num_groups=10):
    random.shuffle(image_identifiers)
    groups = [image_identifiers[i:i + group_size] for i in range(0, len(image_identifiers), group_size)][:num_groups]
    return groups


analogies_groups = create_random_groups(image_identifiers)

print(analogies_groups)


# # Function to create pairs of images (target, reference)
# def create_image_pairs(analogies_groups):
#     image_pairs = []
#     for group in analogies_groups:
#         # Ensure group has enough images
#         if len(group) < 6:
#             continue
#         reference_image = group[-1]  # Last image in the group is the reference
#         for target_image in group[:-1]:  # Iterate over all but the reference image
#             pair = (yaml_data.get(target_image, None), yaml_data.get(reference_image, None))
#             if None not in pair:  # Only add the pair if both images are found in the YAML data
#                 image_pairs.append(pair)
#     return image_pairs


# # Generate and save image pairs
# image_pairs = create_image_pairs(analogies_groups)

# print(image_pairs)

# This is a mockup example of what the analogies data might look like
# Each sub-array represents a group of related images, possibly an analogy set.
# The indices here are imaginary and would correspond to actual image file identifiers.
# analogies_data = np.array([
#     [0, 1, 2, 3, 4, 9],  # a group where image 9 is the reference
#     [10, 11, 12, 13, 14, 19],  # another group with image 19 as the reference
#     # ... more groups
# ])

## NEW VER WITH SIMAILAITY
def load_vgg19_gray_model():
    # Load the VGG-19 model pre-trained on ImageNet
    model = models.vgg19(pretrained=True)
    # Modify the first layer to accept grayscale input, if necessary
    # model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.eval()  # Set the model to evaluation mode
    return model


def extract_features(model, image_path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("L")  # Convert image to grayscale
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        features = model(image)
    return features


def compute_similarity(feature1, feature2):
    # Assuming feature1 and feature2 are numpy arrays
    return 1 - cosine(feature1, feature2)

def generate_reference_pairs(images_dir, model):
    # Your logic here to implement steps 2 to 4
    pass





# Save this array to a '.npy' file
np.save('.\dataset\coco-2017\\analogies_new.npy', analogies_groups)

# # Save this array to a '.npy' file
# np.save('analogies.npy', analogies_data)


# # Main
# if __name__ == "__main__":
#     model = load_vgg19_gray_model()
#     images_dir = './dataset/coco-2017'  # Example path, adjust as needed
#     generate_reference_pairs(images_dir, model)