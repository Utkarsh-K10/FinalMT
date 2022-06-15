import splitfolders
input_folder = 'train'
output = 'dataset'
splitfolders.ratio(input_folder, output = output, seed = 37, ratio = (.8,.2), move =True)
print("DataSet Splitted successfully..")
