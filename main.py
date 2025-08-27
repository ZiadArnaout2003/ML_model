import logging
from fastapi import BackgroundTasks, FastAPI, Request

from . import datamodel
from . import relaxml


app = FastAPI()
relax = relaxml.RelaxML()


@app.on_event('startup')
def startup():
    logging.basicConfig(format='%(levelname)s:\t %(message)s', level=logging.INFO)


@app.post('/predict')
async def predict(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    tasks = [datamodel.Task(t) for t in data.get('tasks', [])]
    background_tasks.add_task(relax.predict, tasks)
    return []


@app.get('/')
@app.get('/health')
def health():
    return {
        'status': 'UP',
        'v2': False
    }


@app.post('/setup')
def setup(data: datamodel.Setup):
    relax.setup(data)
    return { 'model_version': relax.model_version }
 
