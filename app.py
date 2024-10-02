from flask import Flask, render_template, request, redirect, url_for
import cv2
import os

app = Flask(__name__)

# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save the uploaded file to a temporary location
    filepath = os.path.join('static', file.filename)
    file.save(filepath)

    # Load the image
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save the output image
    output_filepath = os.path.join('static', 'output_' + file.filename)
    cv2.imwrite(output_filepath, image)

    return render_template('index.html', output_image=output_filepath)

if __name__ == '__main__':
    app.run(debug=True)
