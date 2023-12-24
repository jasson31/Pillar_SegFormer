import shutil

import cv2
import numpy as np
import os
import random

# Color palette for labeled image
pillar_colors = [[255, 105, 180], [255, 0, 0], [255, 140, 0], [255, 255, 0],
                 [0, 128, 0], [0, 0, 255], [75, 0, 130], [128, 0, 128]]

# Root directory of dataset
datasetRootDir = 'data/pillar_raw'

# Create dataset with given images
def CreateDataset(datasetName, datasetList):
    os.makedirs(f'{datasetRootDir}/../results/images/{datasetName}')
    os.makedirs(f'{datasetRootDir}/../results/annotations/{datasetName}')
    for debugIndex, imageName in enumerate(datasetList):

        print(f'Creating {imageName} : {debugIndex} / {len(datasetList)}')

        # Read current label image
        labelImage = cv2.imread(f'{datasetRootDir}/{imageName}')
        resultImage = np.zeros((360, 640, 1), dtype=np.uint8)

        for i in range(len(pillar_colors)):
            color = np.array(pillar_colors[i][::-1])

            newColor = np.array([0])
            if i == 0 or i == 1:
                newColor = np.array([1])
            else:
                newColor = np.array([2])

            resultImage[cv2.inRange(labelImage, color, color) > 0] = newColor


        shutil.copy(f'{datasetRootDir}/{imageName}'.replace('label', 'camera').replace('png', 'jpg'), f'{datasetRootDir}/../results/images/{datasetName}/{debugIndex}.jpg')
        cv2.imwrite(f'{datasetRootDir}/../results/annotations/{datasetName}/{debugIndex}.png', resultImage)


if __name__ == "__main__":

    # List all sequences
    sequenceList = os.listdir(datasetRootDir)

    totalImageList = []
    for _, sequence in enumerate(sequenceList):
        imageList = os.listdir(f'{datasetRootDir}/{sequence}/label')
        for _, image in enumerate(imageList):
            totalImageList.append(f'{sequence}/label/{image}')

    random.shuffle(totalImageList)

    # Use 3/4 of dataset as training set, 1/4 as validation set
    trainSet = totalImageList[:int(len(totalImageList) * 0.95)]
    validSet = totalImageList[int(len(totalImageList) * 0.95):]

    print(f'Total number : {len(totalImageList)}')
    print(f'Train set : {len(trainSet)}')
    print(f'Valid set : {len(validSet)}')

    CreateDataset('training', trainSet)
    CreateDataset('validation', validSet)

    '''# test set
    testSet = totalImageList

    print(f'Total number : {len(testSet)}')

    CreateDataset('test.json', testSet)'''
