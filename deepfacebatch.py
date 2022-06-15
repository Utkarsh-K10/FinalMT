# import deepface
# from deepface import DeepFace
# models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
# DeepFace.stream('/Users/utkarshkushwaha/Downloads/ITDEPT/testproject/datab', model_name = models[0])

# #1. Face Recognition with Image capture
# #2. FAce recognition wiith Template and time 
# #3. FAce Recognition with Live cam and Details
from operator import indexOf
import pandas as pd
empl = pd.read_csv("/Users/utkarshkushwaha/Downloads/ITDEPT/testproject/cv2user.csv")
lbls = list(empl['fName'].values)
print(type(lbls),lbls)

employee_name = 'Utkarsh_Kushwaha1.jpg'
tname = employee_name.split('_')[0]
print(tname)

if tname in lbls:
    print(tname)
    detail = (empl.loc[[lbls.index(tname)]]).values.tolist()
    print(detail[0])
for itr, word in enumerate(detail[0]):
    print(str(word))


cap = cv2.VideoCapture(0)
while(True):
    ret, img =  cap.read()
