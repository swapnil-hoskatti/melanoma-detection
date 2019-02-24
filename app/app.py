# imports - standard imports
import json
import os
import warnings

# imports - third party imports
from flask import Flask, flash, redirect
from flask import render_template as render
from flask import request, url_for
from werkzeug.utils import secure_filename

# imports - module imports
from core.melancholic import main_app as melancholic

warnings.filterwarnings("ignore")

# setting up constants
UPLOAD_FOLDER = "../temp_files/"
ALLOWED_EXTENSIONS = set(
    [
        "jpg",
        "jpeg",
        "jp2",
        "jpe",
        "sr",
        "ras",
        "pbm",
        "pgm",
        "ppm",
        "png",
        "gif",
        "tif",
        "tiff",
        "bmp",
    ]
)

# setting up Flask instance
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config.from_mapping(SECRET_KEY="dev")

# listing views
link = {}
link["index"] = "/"

if not os.path.exists("../temp_files"):
    os.makedirs("../temp_files")


def allowed_file(filename):
    # check if extension exists & check if extension is allowed
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(url_for("upload_file"))

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = f"OG.{filename.rsplit('.', 1)[1].lower()}"
            current_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            file.save(current_file_path)

            # send to base-app here
            prediction = melancholic(current_file_path)

            # prediction = 'true'

            return render("page.html", result=prediction, upload="true")

    return render("page.html", upload="false")
