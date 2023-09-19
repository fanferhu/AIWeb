import base64
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image 
import torch 
from transformers import BlipProcessor, BlipForConditionalGeneration,BertTokenizer
from model import MyModel
from config import parsers
import io

app = Flask(__name__, static_folder='static') 
app.config['UPLOAD_FOLDER'] = 'static/images'

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model.to(device,torch.float16)
args = parsers()

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load(model_path))
    myModel.eval()
    return myModel

def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


def text_class_name(pred,text):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    print(f"文本：{text}\t预测的类别为:{classification_dict[result[0]]}")
    return classification_dict[result[0]]

bert_model = load_model(device, args.save_model_best)
bert_model.to(device)

@app.route('/') 
def home(): 
 return render_template('home.html')

@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('images')
    filenames = []
    print(files)
    for file in files:
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return redirect(url_for('predict', filenames=filenames))

@app.route('/predict') 
def predict():
    images = []
    filenames = request.args.getlist('filenames')
    print(filenames)
    for filename in filenames:
        image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image = image.resize((224, 224)).convert('RGB')
        images.append(image)
    # Process image and make prediction 
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
    outputs = blip_model.generate(**inputs)
    predictions = []
    for i in range(len(outputs)):
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[i].strip()
        predictions.append(generated_text)
    predictions = ",".join(predictions)
    x = process_text(predictions, args.bert_pred)
    with torch.no_grad():
        pred = bert_model(x)
        classifications = text_class_name(pred,predictions)
    # Render HTML page with prediction results 
    return render_template('predict.html', predictions = predictions, classifications = classifications, filenames = filenames)
	
if __name__ == '__main__':
    app.run()