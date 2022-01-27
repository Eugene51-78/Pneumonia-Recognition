from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
import uvicorn

app = FastAPI()

def predict(img, model):
    predicted = model.predict(img)
    conf = predicted.tolist()[0]
    conf_bacteria = round(conf[0], 8)
    conf_normal = round(conf[1], 8)
    conf_virus = round(conf[2], 8)
    confidence =  {"Bacteria":conf_bacteria, "Normal":conf_normal, "Virus":conf_virus}
    tp = np.argmax(predicted)
    tp = ['Bacteria', 'Normal', 'Virus'][tp]
    return [{"Predict": tp, "Confidence": confidence}]

model = None

@app.get("/", response_class=ORJSONResponse)
def main():
	info = "This application detects the presence of a disease by analyzing a snapshot of the lungs"
	return [{"info": info}]

@app.post("/classify", response_class=ORJSONResponse)
async def classify(img : UploadFile = File(...)):
    # prediction = predict(await img.read())
    global model
    datas = await img.read()
    print(type(datas))
    nparr = np.fromstring(datas, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(img_np.shape)
    img_np = cv2.resize(img_np, (224, 224))
    img_np = np.reshape(img_np, [1,224,224,3])
    img_np = img_np/255.
    if model == None:
        model = open_model()
    prediction = predict(img_np, model)
    return prediction

def open_model():
    json_file = open('app/model_1.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load weights into new model
    model.load_weights('app/model_1.h5')
    print("Model is loaded")
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy", tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model

#if __name__ == "__main__":
    #uvicorn.run("main:app", host="127.0.0.1", port=5000, #log_level="info", reload=True)
