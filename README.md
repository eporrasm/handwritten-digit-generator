# handwritten-digit-generator
Problem 3 (Handwritten Digit Generation Web App)
Overview
Build a web app that can generate images of handwritten digits 0–9. The user should specify which digit.  Then, the app generates 5 images and displays them in a format similar to the MNIST dataset.

Please submit the code file even if it's incomplete. If you have nothing to submit, then you don't need to.

Example


1. Web Application
Requirement
Details
Framework
Implement a web application utilizing a framework such as Streamlit, or alternatively, another suitable web application deployment framework. Any framework or service is fine.
Public Access
Anyone should be able to access the app for up to two weeks. Do not disable it within two weeks of this exam. It is acceptable if the app goes into sleep mode when cold, but ensure external users can reactivate it.
Features
Allow users to select which digit (0-9) to generate.
Generate 5 images of the same digit based on the previous user’s selections,
generate with your own trained model.
Display the 5 images of the same digit.



2. Model Building and Training
Requirement
Details
Dataset
Use the MNIST dataset. (28x28 grayscale)
Framework
Use PyTorch or TensorFlow to train a model from scratch.  

Do NOT use pre trained model weights.   
Training Environment
As a rule, use Google Colab with one T4 GPU on your personal Google account for all training. If you are likely to run out of free Colab resources, use another account (for example, a school account). If you have a paid Colab plan or, also use one T4 GPU. Do NOT use other cloud GPUs with higher performance than one T4 GPU. This is cheating.
Accuracy and Training Time
Accuracy is NOT very critical; as long as ChatGPT-4o can recognize all the generated 5 digits correctly, it’s fine. Generating the exact same image five times is NOT acceptable. However, you may generate similar images with little diversity. You are responsible for deciding an appropriate training time and the image generation method. This is very important.
Training Script
Include the model architecture and loss function.  

3. Submission
Item
Format/Example
Web App Link
https:/...
Training Script
A File containing the model architecture and loss function (.py or .ipynb)


Please submit the training script, even if it's incomplete. If you have nothing to submit, then you don't need to.

