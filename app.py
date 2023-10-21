from flask import Flask, g
from flask_migrate import Migrate
import logging

import config
from extensions import db, mail
import argparse
import os
import torch

from blueprints.detect import bp as detect_bp

app = Flask(__name__)
app.config.from_object(config)

db.init_app(app)
mail.init_app(app)
'''
flask db init
flask db migrate
flask db upgrade
'''
migrate = Migrate(app, db)

app.register_blueprint(detect_bp, url_prefix='/detect')


def init_model(repo_dir, model_path, source='local', device='cpu'):
    if source != 'local':
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
        return model
    model = torch.hub.load(repo_dir, 'custom', path=model_path, source=source, device=device)
    model.eval()
    return model


@app.before_request
def before_request():
    g.repo_dir = repo_dir
    g.weights_path = weights_path
    g.model_path = model_path
    g.model = model
    # setattr(g, 'repo_dir', repo_dir)
    # setattr(g, 'weights_path', weights_path)
    # setattr(g, 'model_path', model_path)
    # setattr(g, 'model', model)


if __name__ == "__main__":
    repo_dir = os.getcwd()
    weights_path = 'weights/yolov5-7.0/yolov5s6.pt'
    # weights_path = 'weights/yolov5-3.1/TACO_yolov5s_300_epochs.pt'
    model_path = os.path.join(repo_dir, weights_path)

    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5003, type=int, help="port number")
    args = parser.parse_args()

    model = init_model(repo_dir, model_path)

    logging.getLogger().setLevel(logging.INFO)
    logging.info('项目已启动')
    logging.info(f'当前工作目录: {repo_dir}')
    logging.info(f'当前调用权重: {weights_path}')
    logging.info(f'模型推断请访问: http://localhost:{args.port}/detect/upload')

    app.run(host="0.0.0.0", port=args.port, debug=True)
