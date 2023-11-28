from flask import Flask, render_template, url_for, request, redirect, flash, jsonify
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import json
import os
import openai
import datetime

app = Flask(__name__)
app.secret_key = 'secret'

disease_type = open("type_disease_detect.json")
disease_type_data = json.load(disease_type)
disease_type_data = disease_type_data["type_disease"]

disease_image = open("disease_image.json")
disease_image_data = json.load(disease_image)
disease_image_data = disease_image_data["images"]

def setup_upload_folder():
    path = os.getcwd()
    UPLOAD_FOLDER = os.path.join(path, 'uploads')
    if not os.path.isdir(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    return UPLOAD_FOLDER


UPLOAD_FOLDER = setup_upload_folder()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


############################### CHAT ############################################


openai.api_key = open("key.txt", "r").read().strip("\n")

message_history = []



# def chat(user_input, role="user"):
#     message_history.append({"role" :  role, "content" : user_input + ';' + datetime.datetime.now().strftime("%H:%M")})
#     completion = openai.ChatCompletion.create(
#     model = "gpt-3.5-turbo-0301",
#     messages = message_history
#     )

#     reply_content = completion.choices[0].message.content
#     print(reply_content)
#     message_history.append({"role" : "assistant", "content" : reply_content + ';' + datetime.datetime.now().strftime("%H:%M")})
#     print(message_history)
#     return reply_content, message_history

def chat(user_input, role="user"):
    message_history.append({"role": role, "content": user_input + ';' + datetime.datetime.now().strftime("%H:%M")})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_history,
        stop=None  # Allow the model to generate a continuous conversation
    )

    reply_content = completion['choices'][0]['message']['content']
    message_history.append({"role": "assistant", "content": reply_content + ';' + datetime.datetime.now().strftime("%H:%M")})
    return reply_content, message_history


@app.route("/")
def home():
    return render_template('home.html', disease_type_data=disease_type_data, disease_image_data=disease_image_data)


@app.route("/upload_image")
def upload_file():
    return render_template('uploadfile.html', title="Upload Image")


@app.route("/submit", methods=['GET', 'POST'])
def submit_file():
    if request.method == 'POST':
        # if 'image' not in request.files:
        #     return 'No file provided', 400
        
        image = request.files['myFile']
        flag = 0
        
        if image and allowed_file(image.filename):
            filename = image.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.seek(0)
            image.save(file_path)
            files = {'myFile':open(file_path, 'rb')}
            pred = prediction(file_path)
            disease_name = disease_info['disease_name'][pred]
            description =disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            if "Healthy" in disease_name: flag = 1
            return render_template('airesult.html', disease_name = disease_name , desc = description , prevent = prevent , flag=flag,
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
        else:
            return render_template('uploadfile.html', message='Invalid file format')
        

# @app.route('/ques_ans')
# def ask_ques():
#     return render_template('ques_ans.html', title="F.A.Q")


# @app.route('/submit_comment', methods=['POST'])
# def get_ques():
#     my_input = request.form.get('my_input')
#     reply_ans, message_history = chat(my_input)
#     return render_template('ques_ans.html', title="F.A.Q", chat_data = message_history)


@app.route('/ques_ans')
def ask_ques():
    return render_template('ques_ans.html', title="F.A.Q")


@app.route('/submit_comment', methods=['POST'])
def get_ques():
    my_input = request.form.get('my_input')
    reply_ans, message_history = chat(my_input)
    return render_template('ques_ans.html', title="F.A.Q", chat_data=message_history)




@app.route('/admin_panel')
def admin_panel_users():
    return render_template('admin_panel.html', title="Admin Panel")


@app.route('/model_charts')
def chart_for_model():
    return render_template('chart_panel.html', title='Charts')


if __name__ == "__main__":
    app.run(
            debug=True,
            port=8000
        )