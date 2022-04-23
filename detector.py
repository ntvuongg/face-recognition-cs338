import cv2
from facenet_pytorch import MTCNN
import torch
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import  Model
from scipy.spatial import distance
from PIL import Image
import numpy as np

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs = vgg16_model.get_layer("fc1").output)
    return extract_model

# Image Preprocessing, image to tensor
def image_preprocess(img):
    img = img.resize((224,224)) # VGG16 constraint
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_vector(model, img):
    # img = Image.open(image_path)
    img_tensor = image_preprocess(img)

    # Features extraction
    vector = model.predict(img_tensor)[0]
    # Vector normalization
    vector = vector / np.linalg.norm(vector)
    return vector

model = get_extract_model()

mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

thanh = extract_vector(model, Image.open('faces/2.jpg'))
vuong = extract_vector(model, Image.open('faces/1.jpg'))
tung = extract_vector(model, Image.open('faces/0.jpg'))
vectors = np.array([tung, vuong, thanh])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

THRESHOLD = 75

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, _ = mtcnn.detect(frame)
        if boxes is not None:
            for box in boxes:
                bbox = list(map(int,box.tolist()))
            
            frame_copy = frame.copy()
            frame_RGB = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            frame_copy = Image.fromarray(frame_RGB)
            # cv2.imwrite('frame.png', frame)
            # thanh_img = Image.open('frame.png')
            crop_img = frame_copy.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
            # crop_img.save('frame.png')
            search_vector = extract_vector(model, crop_img)
            # distance = np.linalg.norm(vectors - search_vector, axis=1)
            idx = []
            for person in vectors:
                idx.append(1 - distance.cosine(search_vector, person))
            idx = np.array(idx)
            label = np.argmax(idx)
            conf = round(idx[label]*100,2)
            if label == 0:
                text = 'Tung'
            elif label == 1:
                text = 'Vuong'
            elif label == 2:
                text = 'Thanh'

            if conf < THRESHOLD:
                text = 'Unkown'
                cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
                cv2.putText(frame, f'{text}', (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            else:
                cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),6)
                cv2.putText(frame, f'{text} ({conf})', (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
                
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1)&0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()