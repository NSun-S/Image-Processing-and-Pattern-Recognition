import scipy.io as scipy
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
from scipy.ndimage.interpolation import shift

'''
    __header__
    __version__
    __globals__
    isTest: test image mark, 0 means train set image, 1 means test Set image
    fea: image
    gnd: ground truth
'''

def shift_images(images):
    shifted_images = []
    for index in range(len(images)):
        dx = np.random.randint(-2,2)
        dy = np.random.randint(-2,2)
        shifted_images.append(shift(images[index],[dx,dy],cval=0,mode='constant'))
    return np.array(shifted_images)

def loadSVMImage():
    pose_05 = scipy.loadmat('../PIE_dataset/Pose05_64x64.mat')
    pose_07 = scipy.loadmat('../PIE_dataset/Pose07_64x64.mat')
    pose_09 = scipy.loadmat('../PIE_dataset/Pose09_64x64.mat')
    pose_27 = scipy.loadmat('../PIE_dataset/Pose27_64x64.mat')
    pose_29 = scipy.loadmat('../PIE_dataset/Pose29_64x64.mat')
    trainSet_x = []
    trainSet_y = []
    testSet_x = []
    testSet_y =[]
    for i in range(len(pose_05['isTest'])):
        if(pose_05['isTest'][i]==1):
            testSet_x.append(pose_05['fea'][i])
            testSet_y.append(int(pose_05['gnd'][i])-1)
        else:
            trainSet_x.append(pose_05['fea'][i])
            trainSet_y.append(int(pose_05['gnd'][i])-1)
    for i in range(len(pose_07['isTest'])):
        if(pose_07['isTest'][i]==1):
            testSet_x.append(pose_07['fea'][i])
            testSet_y.append(int(pose_07['gnd'][i])-1)
        else:
            trainSet_x.append(pose_07['fea'][i])
            trainSet_y.append(int(pose_07['gnd'][i])-1)
    for i in range(len(pose_09['isTest'])):
        if(pose_09['isTest'][i]==1):
            testSet_x.append(pose_09['fea'][i])
            testSet_y.append(int(pose_09['gnd'][i])-1)
        else:
            trainSet_x.append(pose_09['fea'][i])
            trainSet_y.append(int(pose_09['gnd'][i])-1)
    for i in range(len(pose_27['isTest'])):
        if(pose_27['isTest'][i]==1):
            testSet_x.append(pose_27['fea'][i])
            testSet_y.append(int(pose_27['gnd'][i])-1)
        else:
            trainSet_x.append(pose_27['fea'][i])
            trainSet_y.append(int(pose_27['gnd'][i])-1)
    for i in range(len(pose_29['isTest'])):
        if(pose_29['isTest'][i]==1):
            testSet_x.append(pose_29['fea'][i])
            testSet_y.append(int(pose_29['gnd'][i])-1)
        else:
            trainSet_x.append(pose_29['fea'][i])
            trainSet_y.append(int(pose_29['gnd'][i])-1)

    return trainSet_x, trainSet_y, testSet_x, testSet_y

def loadTrainImage(radio, singleGesture=True):
    trainSet_x = []
    trainSet_y = []
    testSet_x = []
    testSet_y =[]
    pose_05 = scipy.loadmat('../PIE_dataset/Pose05_64x64.mat')
    for i in range(len(pose_05['isTest'])):
        if(pose_05['isTest'][i]==1):
            testSet_x.append(np.array(pose_05['fea'][i]).reshape(64,64))
            testSet_y.append(int(pose_05['gnd'][i])-1)
        else:
            trainSet_x.append(np.array(pose_05['fea'][i]).reshape(64,64))
            trainSet_y.append(int(pose_05['gnd'][i])-1)
    if not singleGesture:
        pose_07 = scipy.loadmat('../PIE_dataset/Pose07_64x64.mat')
        pose_09 = scipy.loadmat('../PIE_dataset/Pose09_64x64.mat')
        pose_27 = scipy.loadmat('../PIE_dataset/Pose27_64x64.mat')
        pose_29 = scipy.loadmat('../PIE_dataset/Pose29_64x64.mat')
        for i in range(len(pose_07['isTest'])):
            if(pose_07['isTest'][i]==1):
                testSet_x.append(np.array(pose_07['fea'][i]).reshape(64,64))
                testSet_y.append(int(pose_07['gnd'][i])-1)
            else:
                trainSet_x.append(np.array(pose_07['fea'][i]).reshape(64,64))
                trainSet_y.append(int(pose_07['gnd'][i])-1)
        for i in range(len(pose_09['isTest'])):
            if(pose_09['isTest'][i]==1):
                testSet_x.append(np.array(pose_09['fea'][i]).reshape(64,64))
                testSet_y.append(int(pose_09['gnd'][i])-1)
            else:
                trainSet_x.append(np.array(pose_09['fea'][i]).reshape(64,64))
                trainSet_y.append(int(pose_09['gnd'][i])-1)
        for i in range(len(pose_27['isTest'])):
            if(pose_27['isTest'][i]==1):
                testSet_x.append(np.array(pose_27['fea'][i]).reshape(64,64))
                testSet_y.append(int(pose_27['gnd'][i])-1)
            else:
                trainSet_x.append(np.array(pose_27['fea'][i]).reshape(64,64))
                trainSet_y.append(int(pose_27['gnd'][i])-1)
        for i in range(len(pose_29['isTest'])):
            if(pose_29['isTest'][i]==1):
                testSet_x.append(np.array(pose_29['fea'][i]).reshape(64,64))
                testSet_y.append(int(pose_29['gnd'][i])-1)
            else:
                trainSet_x.append(np.array(pose_29['fea'][i]).reshape(64,64))
                trainSet_y.append(int(pose_29['gnd'][i])-1)
        
        
    trainSet_x = np.array(trainSet_x)
    testSet_x = np.array(testSet_x)
    trainSet_y = np.eye(68)[trainSet_y]
    testSet_y = np.eye(68)[testSet_y]


    size = trainSet_x.shape[0]
    print(size)
    ran = np.random.permutation(size)
    validationSize = int(len(trainSet_x)*radio)

    return (trainSet_x[ran[validationSize:],], trainSet_y[ran[validationSize:],],
            trainSet_x[ran[:validationSize],], trainSet_y[ran[:validationSize],],
            testSet_x, testSet_y)
'''
    print(testSet_x[70], testSet_y[70])
    print(len(testSet_x),len(testSet_y))
    print(len(trainSet_x), len(trainSet_y))
    # print()

    # tmpImg = Image.fromarray(np.array(testSet[1][0]).reshape(64,64))
    # tmpImg = np.array(testSet[1][0]).reshape(64,64)
    plt.imshow(testSet_x[1])
    plt.axis('off')
    plt.show()
'''