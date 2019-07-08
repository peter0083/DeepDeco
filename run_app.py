#!/usr/bin/env python3

from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
from webapp.run_engine_webapp import StyleSearch
import time
from datetime import datetime, timezone
import os
import boto3
import imageio

UPLOAD_FOLDER = 'webapp/upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/image', methods=['POST'])
def hello_world():
    if request.method == 'POST':

        # Part I
        # request checking
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'

        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Part 2
        # Style Search Engine

        print("Style Search Engine")

        user_text = request.form.get('text')

        print("style description: ", user_text)

        start = time.time()

        ss = StyleSearch()
        similarity_score, style_image_name = ss.find_max_similarity(user_text)

        end = time.time()
        style_search_time = end - start

        # Part 3
        # Fast deep photo style transfer

        print("Fast deep photo style transfer inference")

        currentDT = datetime.now(timezone.utc)
        file_name_stamp = currentDT.strftime('%Y_%m_%d_%H_%M_%S')
        print(file_name_stamp)
        print(type(file_name_stamp))

        if request.form.get('speed') == 'slow':
            timer = '3600'  # 60 sec/min * 60 min = 3600 sec

        elif request.form.get('speed') == 'medium':
            timer = '1200'  # 60 sec/min * 20 min = 1200 sec

        else:
            timer = '600'

        print("Style transfer mode: ", request.form.get('speed'), ". timer is", timer, "sec")

        bashCommand40 = "timeout " + timer + " " \
                                             "python src/ftdeepphoto/run_fpst.py --in-path " \
                        + os.path.join(app.config['UPLOAD_FOLDER'], filename) + " " \
                                                                                "--style-path " \
                                                                                "data/" + style_image_name + " --checkpoint-path checkpoints/ --out-path " \
                                                                                                             "output/output_stylized_image" \
                        + file_name_stamp + ".jpg --deeplab-path " \
                                            "src/ftdeepphoto/deeplab/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz " \
                                            "--slow"

        print(bashCommand40)

        start = time.time()
        # os.system(bashCommand40)
        end = time.time()
        style_transfer_time = end - start

        # make a gif
        images = []

        os.system("mkdir -p " + "output/output" + file_name_stamp)

        for file in os.listdir("."):
            if file.endswith(".png"):
                images.append(imageio.imread(file))
        imageio.mimsave("output/output" + file_name_stamp + "/output_stylized_image" + file_name_stamp + ".gif", images)

        print("output/output" + file_name_stamp + "/output_stylized_image" + file_name_stamp + ".gif created")

        # Part 4
        # send file to user

        final_output_img = "output/output" + file_name_stamp + "/output_stylized_image" + file_name_stamp + ".gif"

        return send_file(final_output_img)
