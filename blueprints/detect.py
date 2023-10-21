import datetime
import io
import os

from PIL import Image
from flask import Blueprint, request, render_template, g, redirect
from utils.backend_utils.dir_utils import *

bp = Blueprint(name='detect', import_name=__name__)

DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"


@bp.route("/upload", methods=["GET", "POST"])
def detect():
    model = g.model

    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if not file:
            return

        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_save_name = f"static/detect_result/{now_time}.png"
        output_dir = os.path.join(g.repo_dir, 'static/detect_result')
        create_dir(output_dir)
        Image.fromarray(results.imgs[0]).save(img_save_name)
        return redirect('/' + img_save_name)

    return render_template("index.html")
