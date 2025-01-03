# Automatic-Image-Segmentation-and-Captioning-for-Automatically-Generating-Captions-in-Gujarati
**Generating Captions and Descriptions in Gujarati**

**Overview**

This project integrates cutting-edge AI technologies to perform automatic image segmentation and captioning, focusing on providing descriptive captions in Gujarati. It combines semantic segmentation for object detection, advanced image captioning for English text generation, and neural machine translation to generate Gujarati descriptions.

**Features**

Image Segmentation: Semantic segmentation using DeepLabV3 (ResNet101).
Caption Generation: Contextually accurate captions using the BLIP model.
Gujarati Translation: Neural machine translation for generating captions in Gujarati.
Interactive Output: Allows users to crop specific regions and view Gujarati captions alongside segmented images.

**Technologies Used**

Programming Language: Python
Libraries and Frameworks:
  PyTorch for segmentation and deep learning
  Hugging Face Transformers for BLIP-based captioning
  Google Translate API for Gujarati translation
  PIL and Matplotlib for image processing and interaction
  
**System Workflow**

Input: Users upload an image.
Segmentation: The system identifies objects using DeepLabV3.
Interactive Cropping:** Users select regions of interest.
Caption Generation: Captions are generated for cropped regions using the BLIP model.
Translation: Captions are translated to Gujarati using Google Translate API.
Output: The segmented image and Gujarati captions are displayed.

**Project Architecture**

Segmentation: DeepLabV3 for semantic segmentation.
Captioning: BLIP (Bootstrapped Language-Image Pretraining) for generating English captions.
Translation: Google Translate API for converting captions to Gujarati.

**Results**

Segmentation Accuracy: Achieved >90% Intersection over Union (IoU) for simple images.
Caption Relevance: BLEU-1 score of 84% and BLEU-4 score of 73%.
Translation Effectiveness: Average score of 8.6/10 based on native speaker evaluation.
Future Scope

Improved Segmentation: Explore alternative segmentation techniques for greater accuracy.
Translation Refinement: Fine-tune models for Gujarati to handle complex linguistic nuances.
Real-Time Application: Develop a lightweight mobile app for real-time processing.
Multi-Language Support: Extend support to other regional languages like Hindi and Tamil.
Contribution

Contributions are welcome! Feel free to open issues and submit pull requests.

**License**

This project is licensed under the MIT License.

**Acknowledgments**

DeepLabV3 for segmentation.
BLIP model for image captioning.
Google Translate API for translation.
Pandit Deendayal Energy University for guidance and support.

**Ongoing Work**

I am actively working on enhancing the image segmentation technique to improve accuracy and performance. Future iterations will explore alternative architectures like U-Net, Mask R-CNN, and transformers for semantic segmentation.
