from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from image_recognition import image_recognition
from generate_recipe import generate_recipe
import re

app = Flask(__name__)

# Set up the directory to save uploaded files
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allow only specific file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Recognize the dish from the image
            dish_name = image_recognition(file_path)

            # Generate the recipe for the dish
            recipe = generate_recipe(dish_name)

            recipe = re.sub(r'\*\*(.*?)\*\*', r'<br><strong>\1</strong><br>', recipe)

            recipe = recipe.replace('*', '<br>')
            recipe = recipe[3:]

            return render_template('result.html', dish_name=dish_name, recipe=recipe, image_url=f'uploads/{filename}')

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)