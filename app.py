import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models

# Set page config
st.set_page_config(
    page_title="Emotion Classification",
    layout="centered"
)

# Define the model architecture
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionClassifier, self).__init__()
        self.model = models.resnet18(pretrained=False)
        # Keep original conv1 layer (3 channels)
        num_features = self.model.fc.in_features
        # Create sequential fully connected layer to match saved model
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Load the model
@st.cache_resource
def load_model():
    model = EmotionClassifier()
    model.load_state_dict(torch.load('model_1.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Define image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3 channels
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Define emotion labels
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Create the Streamlit interface
st.title("Emotion Classification App")
st.write("Upload an image to classify the emotion")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process the image and make prediction
    if st.button("Predict Emotion"):
        # Load model
        model = load_model()
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Display results
        st.write(f"Predicted Emotion: {emotion_labels[predicted_class].upper()}")

# Add some information about the app
st.markdown("---")
st.markdown("""
### About
This app uses a ResNet18 model trained on facial emotion recognition.
The model can classify images into 6 different emotions:
- Angry
- Fear
- Happy
- Neutral
- Sad
- Surprise

Upload a clear facial image to get the emotion prediction!
""") 