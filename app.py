# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 11:35:58 2021

@author: ssingh4
"""
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
# ---- coding:utf-8 ----
from flask import Flask, jsonify, request, render_template, redirect, send_from_directory, url_for, send_file
import matplotlib.pyplot as plt
import librosa
import torch
#import IPython.display as ipd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time
import os
from werkzeug.utils import secure_filename
from transformers import Wav2Vec2ProcessorWithLM
from transformers import AutoTokenizer
import csv
import sys
sys.getdefaultencoding()

app = Flask(__name__)
uploads = '/home/s/ssingh4/salp/uploads/'
punjabi = True
maori = False
if punjabi:
    current_asr = 'Punjabi ASR'
    print("************Punjabi ASR********")
    model = Wav2Vec2ForCTC.from_pretrained("./xlsr-punjabi-asr/runs/all-pl-itr3-wav2vec2-large-xlsr-53/checkpoint-31000").to("cuda")
    processor = Wav2Vec2Processor.from_pretrained("./xlsr-punjabi-asr")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("./xlsr-punjabi-asr/")
    vocab_dict = tokenizer.get_vocab()
    sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    print(list(sorted_vocab_dict.keys()))

    from pyctcdecode import build_ctcdecoder

    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path="./5gram.arpa",
        alpha=0.7,
        beta=4.0
    )
if maori:
    current_asr = 'Māori ASR'
    print("************Maori ASR********")
    model = Wav2Vec2ForCTC.from_pretrained("./xlsr-maori-asr/checkpoint-3000").to("cuda")
    processor = Wav2Vec2Processor.from_pretrained("./xlsr-maori-asr/")
    model.eval()

    
def get_prediction(input_dict):
    with torch.no_grad():
        logits = model(input_dict.input_values.to("cuda")).logits#[0].cpu().numpy()
        #print(logits)
        #librosa.display.waveplot(audio_input, sr=sample_rate)
        pred_ids = torch.argmax(logits, dim=-1)[0]
        pred_str = processor.decode(pred_ids)
        print(pred_str)
        return pred_str

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print(file)
        with open("/home/s/ssingh4/salp/uploads/"+file.filename, 'wb') as audio:
            file.save(audio)
        print('file uploaded successfully')
        
        #filename = secure_filename(file.filename)
        print('file is:',file.filename)
        if  file is None or file.filename == "":
                return jsonify({'Error':'No file!'})
        upload_path = "/home/s/ssingh4/salp/uploads/"+file.filename
        #file.save(upload_path)
        #try:
        print("/home/s/ssingh4/salp/uploads/"+file.filename)
        audio_input, sample_rate = librosa.load("/home/s/ssingh4/salp/uploads/"+file.filename, sr = 16000)
        input_dict = processor(audio_input, sampling_rate = sample_rate, return_tensors="pt", padding=True)
        pred = get_prediction(input_dict)
        with open("/home/s/ssingh4/salp/results.tsv", 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow([file.filename, pred])

        return render_template('index.html', pred=pred, path="/uploads/"+file.filename)
    return render_template('index.html', current_asr=current_asr)

@app.route('/uploads/<filename>' , methods=['GET', 'POST'])
def upload(filename):
    return send_from_directory(uploads, filename)

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    
    with open('feedback.txt', 'a') as file:
        file.write(feedback + '\n')
        
    return '<h1 class="display-4">Thanks for your feedback! </h1> <p> <a href="/" class="badge badge-warning"> Return to home</a>'

@app.route('/data')
def serve_data():
    return send_file('#', mimetype='application/zip')
# if __name__ == '__main__':
#     app.run(debug=True, use_reloader=True, host='0.0.0.0')





# from flask import  Flask, make_response  

# from flask_cors import CORS


# app=Flask(__name__)

# CORS(app, resources=r'/*')

# @app.route('/welcome',methods = ["GET", "POST"])
# def index():
#     return 'welcome to my webpage!'


# @app.route('/hello')
# def hello_world():
#    return 'hello world'

# from flask import Flask, request  
  
  
# @app.route('/1')  
# def home():  
#     response = make_response('Hello, World!')  
#     response.headers['Content-Type'] = 'application/json'
#     return response



# from flask import send_from_directory
# import os
# @app.route('/download', methods = ["GET", "POST"])
# def download():
#     print("GOGOGOGOGOGOGOGOGOGO")
#     print(request.data)
#     response = make_response(
#             send_from_directory(directory='./',path='2021-10-15T04-25-46-345Z.wav')
#             )#将1.jpg（即你的处理结果）以文件流返回
#     # return response
#     return send_from_directory(directory='./',path='2021-10-15T04-25-46-345Z.wav')


if __name__=="__main__":
    app.run(port=8081,host="0.0.0.0",debug=True)






