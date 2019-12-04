import pandas as pd
import cv2
import os
import numpy as np

folder = "D://Biomedical_Tubule//subs"

def load_images_from_folder():
    images = []
    labels = []
    filenames = []

    counter = 1
    for filename in os.listdir(folder):
        label = filename.split("_")[2]
        if label == "anno":
            label = filename.split("_")[3]

        labels.append(label)
        filenames.append(filename)


        img = cv2.imread(os.path.join(folder,filename))

        if counter % 10000 == 0:
            print(counter)
        counter += 1

        if img is not None:
            images.append(img)
            labels.append(label)
            filenames.append(filename)

    return [images,labels,filenames]

data_pickle = load_images_from_folder()


images = np.array(data_pickle[0])
np.save('D://Biomedical_Tubule//images.npy',images)


df = pd.DataFrame(list(zip(data_pickle[1], data_pickle[2])), columns=["label","filename"])
df.to_csv("D://Biomedical_Tubule//label_filename.csv")



