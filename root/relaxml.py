import logging
import os
import requests
from dotenv import load_dotenv
from label_studio_tools.core.label_config import parse_config
from requests.auth import HTTPBasicAuth
from typing import List

from .datamodel import Setup, Task
from .utils import uri_to_url, download_url
from PIL import Image

load_dotenv()
class RelaxML:
    def __init__(self):
        '''Good place to load your model and setup variables'''

        self.project = None
        self.schema = None
        self.hostname = None
        self.access_token = None
 
        self.user = os.getenv("DAGSHUB_USER_NAME")
        self.token = os.getenv("DAGSHUB_TOKEN")
        self.repo = os.getenv("DAGSHUB_REPO_NAME")
        self.owner = os.getenv("DAGSHUB_REPO_OWNER")

        # HERE: Load model


    def setup(self, setup: Setup):
        '''Store the setup information sent by Label Studio to the ML backend'''

        self.project = setup.project
        self.parsed_label_config = parse_config(setup.label_schema)
        self.hostname = setup.hostname
        self.access_token = setup.access_token
        self.model_version = "0.1"

        from_name, schema = list(self.parsed_label_config.items())[0]
        self.from_name = from_name
        self.to_name = schema['to_name'][0]
        self.labels = schema['labels']

    
    def send_predictions(self, result):
        '''Send prediction results to Label Studio'''

        url = f'https://dagshub.com/{self.owner}/{self.repo}/annotations/git/api/predictions/'
        auth = HTTPBasicAuth(self.user, self.token)
        res = requests.post(url, auth=auth, json=result)
        if res.status_code != 200:
            logging.warning(res)
      

    def predict(self, tasks: List[Task]):
        for task in tasks:
            # 1. Get image URI from task
            uri = task.data['image']
            url = uri_to_url(uri, self.owner, self.repo)
            image_path = download_url(url, self.user, self.token)

            # 2. Open image
            img = Image.open(image_path)
            img_w, img_h = img.size

            # 3. Run your model's prediction
            objs = self.model.predict(img)  # Make sure self.model is loaded

            # 4. Process predictions
            lowest_conf = 2.0
            img_results = []

            for obj in objs:
                x, y, w, h, conf, cls = obj
                cls = int(cls)
                conf = float(conf)
                # normalize coordinates to percentage
                x = 100 * float(x - w / 2) / img_w
                y = 100 * float(y - h / 2) / img_h
                w = 100 * float(w) / img_w
                h = 100 * float(h) / img_h

                if conf < lowest_conf:
                    lowest_conf = conf

                label = self.labels[cls]

                img_results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [label],
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h
                    },
                    'score': conf
                })

            result= {
                'task_id': task.id,
                'predictions': img_results,
                'model_version': self.model_version,
            }
            if lowest_conf < 1.0:
                result['score'] = lowest_conf
            # 5. Send predictions to Label Studio
            self.send_predictions(result)
            
