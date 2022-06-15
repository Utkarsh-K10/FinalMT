import Augmentor
import os, random
from Augmentor.ImageUtilities import AugmentorImage

inputpath = '/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train'

for i in os.listdir(inputpath):
    print(f"Inside {i} of Train Directory")
    AugmentorImage.output_directory = f"/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/{i}"
    random.shuffle(os.listdir(f'/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/{i}'))
    p = Augmentor.Pipeline(f"/Users/utkarshkushwaha/Downloads/ITDEPT/Deep-Learning-Face-Recognition-master/Dataset/Train/{i}")
    p.zoom(probability = 0.4)
    p.sample(1)
    if 'output' in os.listdir(f'{inputpath}/{i}/output'):
        os.remove(f'{inputpath}/{i}/output')    
print('sample created successfully')


