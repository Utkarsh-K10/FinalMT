from email import message
import Augmentor
from Augmentor.ImageUtilities import AugmentorImage
import os, random

fpath = "/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Test"
print(os.litdir(fpath))

def Create_smaple(path):

    for i in os.listdir(path):
        p = AugmentorImage.output_directory = f'/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Test/{i}/output'
        p = Augmentor.Pipeline(os.path.join(path,f'{i}'))
        p.zoom(probability= 0.5)
        p.sample(1)
        if 'output' in os.listdir(f'{fpath}/{i}/'):
            os.remove(f'{fpath}/{i}/output')
    message = "sample created successfully.."
    return message

Create_smaple(fpath)



