#Received image file from the input. No validation is done on the input file.
#input imagefile is sent to cnnModel, where it is decoded, resized and sent to model for prediction


from flask import Flask,render_template,request
import os
import numpy as np
from cnnModel import cnnModel
app = Flask(__name__)
@app.route("/", methods=['POST','GET'])
def upload():
  if request.method == 'POST':
    imagefile = request.files['imagefile']
    if imagefile:
      file = imagefile.filename
      imageBytes = imagefile.read()
      model = cnnModel(imageBytes)
      camera = model.predict()
      output = [file,camera]
      return render_template('upload.html',output=output)
    else:
      return render_template('upload.html')
  else:
    return render_template('upload.html')

if __name__ == "__main__":
  app.run(debug=True)