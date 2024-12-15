import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure it loads on the CPU)
DEVICE = torch.device("cpu")  # Force loading the model on CPU
model = models.efficientnet_b0(pretrained=False)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier[1].in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, 17)  # 17 flower classes
)
model.load_state_dict(torch.load('model/best_flower_model.pth', map_location=torch.device('cpu')))
model = model.to(DEVICE)
model.eval()

# Define transforms for input image
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define flower classes
flower_classes = [
    'bluebell', 'buttercup', 'colts_foot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lily_valley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', ' yellow tulip', 'windflower'
]

# Set upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home route (index page)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Open and process the image
        img = Image.open(file_path)
        img = data_transforms(img).unsqueeze(0).to(DEVICE)

        # Make prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted_class = torch.max(outputs, 1)
        
        # Get the predicted class label
        predicted_label = flower_classes[predicted_class.item()]

        # Send the result back to the user with the uploaded image
        return render_template('index.html', filename=filename, label=predicted_label)

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
