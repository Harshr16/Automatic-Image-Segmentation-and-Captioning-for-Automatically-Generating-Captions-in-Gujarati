import torch
from torchvision import models, transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
from googletrans import Translator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

image_path ="/Users/harshrudani/Desktop/Minor Project/eaa638dfee6cdc31c94d8270839d5eb5.jpg"

# Load the segmentation model
def load_segmentation_model(model_name='deeplabv3_resnet101', pretrained=True):
    model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained) if model_name == 'deeplabv3_resnet101' else models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=pretrained)
    model.eval()
    return model

# Segment the image
def segment_image(image_path, model_name='deeplabv3_resnet101', pretrained=True):
    model = load_segmentation_model(model_name, pretrained)
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

# Load the captioning model
def load_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Generate the caption
def generate_caption(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Translate the caption to Gujarati using Google Translate
def translate_to_gujarati(text):
    translator = Translator()
    translation = translator.translate(text, src='en', dest='gu')
    return translation.text


def select_object_in_image(image_path):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title("Draw a rectangle and close the plot")
    print("Click to define two opposite corners of the object in the image")

    # Get user input 
    pts = plt.ginput(2)
    plt.close()

    # Convert points to coordinates
    x1, y1 = int(pts[0][0]), int(pts[0][1])
    x2, y2 = int(pts[1][0]), int(pts[1][1])
    cropped_image = img.crop((x1, y1, x2, y2))
    
    display(cropped_image)  # Display the cropped image for confirmation
    return cropped_image

# Combine segmentation, captioning, and translation
def process_image_for_gujarati_caption(image_path):
    print("Please select the object by drawing a rectangle around it.")
    cropped_image = select_object_in_image(image_path)
    
    if cropped_image:
        processor, caption_model = load_captioning_model()
        caption = generate_caption(cropped_image, processor, caption_model)
        
        gujarati_caption = translate_to_gujarati(caption)
        
        print("Generated Caption in Gujarati:", gujarati_caption)
    else:
        print("No object selected.")

# Example usage
process_image_for_gujarati_caption("/Users/harshrudani/Desktop/Minor Project/eaa638dfee6cdc31c94d8270839d5eb5.jpg")
