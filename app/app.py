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
from . import format_tuple

# ignoring warnings and TF logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# setting up constants
UPLOAD_FOLDER = "../temp_files/"
ALLOWED_EXTENSIONS = set(
	[
		"bmp",
		"dib",
		"jpeg",
		"jpg",
		"jpe",
		"jpeg",
		"jp2",
		"png",
		"pbm",
		"pgm",
		"ppm",
		"tiff",
		"tif",
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
			output_tuple = melancholic(
				current_file_path
			)

			features, features_norm, pred_logistic, pred_cnn = format_tuple(output_tuple)

			return render(
				"page.html",
				features=features,
				features_norm=features_norm,
				pred_logistic=pred_logistic,
				pred_cnn=pred_cnn,
				upload="true",
			)

	return render("page.html", upload="false")
