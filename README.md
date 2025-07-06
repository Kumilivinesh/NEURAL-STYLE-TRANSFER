# NEURAL-STYLE-TRANSFER

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : KUMILI VINESH

*INTERN ID* : CITSOD647

*DOMAIN* : ARTIFICIAL INTELLIGENCE

*DURATION* : 4 WEEKS

*MENTOR* : NEELA SANTOSH

Neural Style Transfer (NST) is a deep learning technique that merges the content of one image with the style of another. Originally introduced by Gatys et al. in 2015, NST leverages convolutional neural networks (CNNs) to separate and recombine image content and artistic style. The core idea is to extract the content features from a source image and the style features from a target artwork, then generate a new image that preserves the content of the first while mimicking the artistic texture of the second.

The process begins by passing the content and style images through a pretrained CNN, typically VGG19, which has been trained on the ImageNet dataset. This model is particularly effective due to its ability to retain hierarchical spatial information through its convolutional layers. In NST, the deeper layers of the CNN are used to capture content (like objects and layout), while the earlier layers are better at capturing style (like colors, brush strokes, and textures).

To perform NST, the algorithm uses a loss function that consists of two parts: content loss and style loss. Content loss measures the difference between the feature representations of the content image and the generated image, ensuring the generated image keeps the structure of the original. Style loss compares the correlations (Gram matrices) of feature activations between the style image and the generated image, helping replicate artistic patterns. These two losses are combined using weight parameters that allow tuning the output toward more content-preserving or style-rich results.

The optimization is usually performed using gradient descent or variants like Adam, updating the pixels of a generated image to minimize the combined loss. Unlike traditional training of neural networks, where weights are updated, NST modifies the input image itself to gradually evolve toward the desired combination of style and content.

In terms of tools, Python is the primary programming language used, with deep learning frameworks such as TensorFlow and PyTorch being the most common choices. PyTorch, in particular, is preferred by many researchers and developers for NST due to its dynamic computational graph and intuitive syntax. The Matplotlib and PIL libraries are typically used for image processing and visualization.

A popular and accessible platform for running Neural Style Transfer experiments is Google Colab. It provides a free cloud-based Jupyter notebook environment with GPU support, which significantly speeds up the processing required for NST. Colab is especially useful for students, hobbyists, and researchers who may not have access to high-performance hardware. By uploading images and writing code in a Colab notebook, users can generate stylized outputs within minutes, thanks to the accelerated computations on GPUs or TPUs.

In summary, Neural Style Transfer is a powerful and creative application of convolutional neural networks, blending artistic expression with modern AI. Using tools like PyTorch, VGG19, and platforms like Google Colab, anyone can explore the fascinating intersection of art and deep learning by turning photos into works of art styled after famous paintings or custom designs.

*output :*
<img width="1440" height="900" alt="Image" src="https://github.com/user-attachments/assets/6d615302-7bba-45c8-995e-697e6fbeb3b6" />
