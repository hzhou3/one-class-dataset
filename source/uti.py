__author__ = 'Hao - 6/2020'

import os
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import math
import random
import time
from dataset3 import fabricSet_One, fabricSet_All, patch_size
from model import autoencoder
from scipy.spatial.distance import pdist, squareform, cdist


# patch_size = 32

'''
overlay results on images
'''

def overlayresults(datasets, id_p, id_n, angl):

    # img = np.zeros((512,512))
    print("number of (val, def) detected as defects: ", len(id_p), len(id_n))

    path = os.path.dirname(os.path.realpath(__file__))
    # path = path.replace("source", "data/rawimage/") + datasets.img_name

    path = path.replace("source", "data/rotationraw_paper/"+str(angl)+"/") + datasets.img_name
    

    raw = Image.open(path)

    img = raw.load()
    

    for i in range(len(id_p)):
        left  = datasets.pos_val_p[id_p[i]][0]
        top   = datasets.pos_val_p[id_p[i]][1]
        right = datasets.pos_val_p[id_p[i]][2]
        btm   = datasets.pos_val_p[id_p[i]][3]
        for y in range(top, btm):
            for x in range(left, right):
                img[y,x] = 1

    for i in range(len(id_n)):
        left  = datasets.pos_n[id_n[i]][0]
        top   = datasets.pos_n[id_n[i]][1]
        right = datasets.pos_n[id_n[i]][2]
        btm   = datasets.pos_n[id_n[i]][3]
        for y in range(top, btm):
            for x in range(left, right):
                img[y,x] = 1
    # raw.show()

    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/" + str(angl) +"/overlay/") 
    if not os.path.exists(path):
        os.makedirs(path)


    raw.save(path + datasets.img_name)


'''
save trained models to .pth
'''

def save_model(filename, model, angl):
    file = filename.split(".")[0]
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/" + str(angl)+"/")
    dirr = path + 'models/'
    if not os.path.exists(dirr):
            os.makedirs(dirr)
    dirr = dirr +file+"/"
    if not os.path.exists(dirr):
            os.makedirs(dirr)

    torch.save(model.state_dict(), dirr+'model.pth')



def test_model(gray=False):
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/")
    

    num_epochs = 20
    batch_size = 128
    learning_rate = 1e-3

    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/rawimage/") + filename

    img = Image.open(path).convert("L")
    img = img.crop((0, 0, patch_size, patch_size))
    # img = img.resize((200, 200))
    data = preprocess(img, gray=gray)

    print("input shape : ", data.shape)
    model = autoencoder(0)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

   
    # ===================forward=====================
    output, code = model(data)
    print("output shape: ", output.shape)
    print("code shape  : ", code.shape)

    loss = criterion(output, data)
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

        
'''
train script
'''
def train(filename, datasets, epoch, angle, gabor=False):


    num_epochs = epoch
    batch_size = 128
    learning_rate = 1e-3
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    model = autoencoder(datasets.dim_in, gabor=gabor)
    criterion = nn.SmoothL1Loss() 
    # criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            # ===================forward=====================
            output, code = model(data)
            # print(data.shape, output.shape, code.shape)
            loss = criterion(output, data)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        # print('epoch [{}/{}], loss:{:.4f}'
        #       .format(epoch+1, num_epochs, loss.data.item()))
        if epoch % 10 == 0 and gabor == False:
            save_batch(output.data, epoch)
            # pic = to_img(output.data)
            # save_image(pic, './dc_img/image_{}.png'.format(epoch))

    save_model(filename, model, angle)



'''
post process. this is when autoencoder is on.
'''
def postprocess_latent_code_gabor(filename, angl, datasets = None, datasets_train=None):

    if datasets == None:
        path = os.path.dirname(os.path.realpath(__file__))
        path = path.replace("source", "data/")
        datasets = fabricSet_One(path, filename, angl, patch_size=patch_size, stride=10, transform = None)

    train_p = len(datasets_train.train_feats)
    val_p = len(datasets.val_feats)
    val_n = len(datasets.neg_feats)

    model = autoencoder(datasets.dim_in, gabor=True)
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/" + str(angl) + "/models/") + filename.split(".")[0] + "/model.pth"
    # print("path is ", path)
    model.load_state_dict(torch.load(path))
    model.eval()

    time1 = time.perf_counter()

    out_train_p = []
    out_val_p = []
    out_val_n = []

    in_train_p = []
    in_val_p = []
    in_val_n = []

    code_train_p = []
    code_val_p = []
    code_n = []



    for i in range(train_p):
        in_train_p.append(datasets_train.train_feats[i])
        img = torch.from_numpy(datasets_train.train_feats[i])
        img = img.unsqueeze(0)
        # print(img.shape)
    
        _, code = model(img)
        code_train_p.append(code.detach().numpy())

    for i in range(val_p):
        in_val_p.append(datasets.val_feats[i])
        img = torch.from_numpy(datasets.val_feats[i])
        img = img.unsqueeze(0)

        _, code = model(img)
        code_val_p.append(code.detach().numpy())

    for i in range(val_n):
        in_val_n.append(datasets.neg_feats[i])
        img = torch.from_numpy(datasets.neg_feats[i])
        img = img.unsqueeze(0)

        _, code = model(img)
        code_n.append(code.detach().numpy())


    # convert code to feature vector (N, 90)

    code_out_train_p = np.stack(code_train_p, axis=0).reshape(-1, 90)
    code_out_val_p = np.stack(code_val_p, axis=0).reshape(-1, 90)
    code_out_n = np.stack(code_n, axis=0).reshape(-1, 90)

    # print("shape of train", code_out_train_p.shape)
    # print("shape of val", code_out_val_p.shape)
    # print("shape of def", code_out_n.shape)

    # C = diag(var(train,1,1));

    # following is how to calculate Nearest neighbor 
    
    train_dist = squareform(pdist(code_out_train_p)) 
    diag = np.ones(train_dist.shape)*1.0e+10
    train_dist += np.diag(np.diag(diag)) #, 'mahalanobis', VI=None
    train_dist = np.amin(train_dist, axis=1)
    # print("pdist of train", train_dist.shape)



    testp_dist = np.amin(cdist(code_out_val_p, code_out_train_p), axis=1)
    nnidp      = np.argmin(cdist(code_out_val_p, code_out_train_p), axis=1)

    testn_dist = np.amin(cdist(code_out_n, code_out_train_p), axis=1)
    nnidn      = np.argmin(cdist(code_out_n, code_out_train_p), axis=1)

    # print("reshape of testp_dist", testp_dist.shape)
    # print("reshape of nnidp", nnidp.shape)

    # print("reshape of testn_dist", testn_dist.shape)
    # print("reshape of nnidn", nnidn.shape)

    labels = np.concatenate((np.ones((testp_dist.shape[0],1)),np.zeros((testn_dist.shape[0],1))), axis=0)
    # print("label shape: ",labels.shape)

    # for i in range(len(nnidp)):
    #     testp_dist[i] = testp_dist[i] / train_dist[nnidp[i]]
    # for i in range(len(nnidn)):
    #     testn_dist[i] = testn_dist[i] / train_dist[nnidn[i]]

    scores = np.concatenate((testp_dist ,testn_dist), axis=0)

    # print("score shape: ",scores.shape)

    maxratio = max(scores[0:testp_dist.shape[0]-1])
    # print("max ratio is: ", maxratio)

    # 1.05 here can be changed but must be greater than 1.0

    pred = scores > maxratio * 1.05

    time2 = time.perf_counter() - time1

    id_p = []
    id_n = []

    # print("FINDING OUTLIERS")

    for i in range(testp_dist.shape[0]):
        if pred[i] == True:
            id_p.append(i)

    
    for i in range(testp_dist.shape[0], len(pred)):
        if pred[i] == True:
            id_n.append(i-testp_dist.shape[0])


    fpr, tpr, thresholds, roc_auc, tpr1 = generateROC(labels, scores, filename, angl)


    normal_idx = np.argwhere(labels == 1)
    # print("normal_idx shape: ", normal_idx.shape)
    FPR = sum(pred[normal_idx])/len(normal_idx)

    defect_idx = np.argwhere(labels == 0)
    # print("defect_idx shape: ", defect_idx.shape)
    TPR = sum(pred[defect_idx])/len(defect_idx)

    # print("FPR, TPR: ", FPR, TPR)

    # identify ids of the image patches in the test data that are predicted
    # as defect
    ID1 = np.argwhere(pred[0:len(datasets.val_feats)] == 1)
    ID2 = np.argwhere(pred[len(datasets.val_feats):len(pred)] == 1)
    # print("shape of ID1, ID2: ", ID1.shape, ID2.shape)


    overlayresults(datasets, id_p, id_n, angl)
    return tpr1, time2


'''
post process. this is when autoencoder is off, that is only mean and std as features.
'''
def postprocess_latent_code_firstorder(filename, angl, datasets = None, datasets_train=None):

    if datasets == None:
        path = os.path.dirname(os.path.realpath(__file__))
        path = path.replace("source", "data/")
        datasets = fabricSet_One(path, filename, angl, patch_size=patch_size, stride=10, transform = None)


    train_p = len(datasets_train.train_feats)
    val_p = len(datasets.val_feats)
    val_n = len(datasets.neg_feats)

    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/" + str(angl) + "/models/") + filename.split(".")[0] + "/model.pth"


    time1 = time.perf_counter()

    out_train_p = []
    out_val_p = []
    out_val_n = []

    in_train_p = []
    in_val_p = []
    in_val_n = []

    code_train_p = []
    code_val_p = []
    code_n = []


    code_out_train_p = datasets_train.train_feats
    code_out_val_p = datasets.val_feats
    code_out_n = datasets.neg_feats

    train_dist = squareform(pdist(code_out_train_p)) 
    diag = np.ones(train_dist.shape)*1.0e+10
    train_dist += np.diag(np.diag(diag)) #, 'mahalanobis', VI=None
    train_dist = np.amin(train_dist, axis=1)


    testp_dist = np.amin(cdist(code_out_val_p, code_out_train_p), axis=1)
    nnidp      = np.argmin(cdist(code_out_val_p, code_out_train_p), axis=1)

    testn_dist = np.amin(cdist(code_out_n, code_out_train_p), axis=1)
    nnidn      = np.argmin(cdist(code_out_n, code_out_train_p), axis=1)

    labels = np.concatenate((np.ones((testp_dist.shape[0],1)),np.zeros((testn_dist.shape[0],1))), axis=0)

    scores = np.concatenate((testp_dist ,testn_dist), axis=0)

    maxratio = max(scores[0:testp_dist.shape[0]-1])

    pred = scores > maxratio * 1.05

    time2 = time.perf_counter() - time1

    id_p = []
    id_n = []

    for i in range(testp_dist.shape[0]):
        if pred[i] == True:
            id_p.append(i)

    
    for i in range(testp_dist.shape[0], len(pred)):
        if pred[i] == True:
            id_n.append(i-testp_dist.shape[0])


    fpr, tpr, thresholds, roc_auc, tpr1 = generateROC(labels, scores, filename, angl)


    normal_idx = np.argwhere(labels == 1)
    FPR = sum(pred[normal_idx])/len(normal_idx)

    defect_idx = np.argwhere(labels == 0)
    TPR = sum(pred[defect_idx])/len(defect_idx)

    # identify ids of the image patches in the test data that are predicted
    # as defect
    ID1 = np.argwhere(pred[0:len(datasets.val_feats)] == 1)
    ID2 = np.argwhere(pred[len(datasets.val_feats):len(pred)] == 1)

    overlayresults(datasets, id_p, id_n, angl)
    return tpr1, time2


'''
post process. this is when PCA is on.
'''
def postprocess_latent_code_pca(filename, angl, datasets = None, datasets_train=None):

    if datasets == None:
        path = os.path.dirname(os.path.realpath(__file__))
        path = path.replace("source", "data/")
        datasets = fabricSet_One(path, filename, angl, patch_size=patch_size, stride=10, transform = None)

    train_p = len(datasets_train.train_feats)
    val_p = len(datasets.val_feats)
    val_n = len(datasets.neg_feats)

    # here is how to apply PCA

    import numpy as np
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 90)
    pca.fit(datasets_train.train_feats)


    time1 = time.perf_counter()

    out_train_p = []
    out_val_p = []
    out_val_n = []

    in_train_p = []
    in_val_p = []
    in_val_n = []

    code_train_p = []
    code_val_p = []
    code_n = []


    code_out_train_p = pca.transform(datasets_train.train_feats)
    code_out_val_p = pca.transform(datasets.val_feats)
    code_out_n = pca.transform(datasets.neg_feats)

    train_dist = squareform(pdist(code_out_train_p)) 
    diag = np.ones(train_dist.shape)*1.0e+10
    train_dist += np.diag(np.diag(diag)) #, 'mahalanobis', VI=None
    train_dist = np.amin(train_dist, axis=1)
    # print("pdist of train", train_dist.shape)



    testp_dist = np.amin(cdist(code_out_val_p, code_out_train_p), axis=1)
    nnidp      = np.argmin(cdist(code_out_val_p, code_out_train_p), axis=1)

    testn_dist = np.amin(cdist(code_out_n, code_out_train_p), axis=1)
    nnidn      = np.argmin(cdist(code_out_n, code_out_train_p), axis=1)

    labels = np.concatenate((np.ones((testp_dist.shape[0],1)),np.zeros((testn_dist.shape[0],1))), axis=0)

    scores = np.concatenate((testp_dist ,testn_dist), axis=0)

    # print("score shape: ",scores.shape)

    maxratio = max(scores[0:testp_dist.shape[0]-1])

    pred = scores > maxratio * 1.0

    time2 = time.perf_counter() - time1

    id_p = []
    id_n = []


    for i in range(testp_dist.shape[0]):
        if pred[i] == True:
            id_p.append(i)

    
    for i in range(testp_dist.shape[0], len(pred)):
        if pred[i] == True:
            id_n.append(i-testp_dist.shape[0])


    fpr, tpr, thresholds, roc_auc, tpr1 = generateROC(labels, scores, filename, angl)


    normal_idx = np.argwhere(labels == 1)
    FPR = sum(pred[normal_idx])/len(normal_idx)
    defect_idx = np.argwhere(labels == 0)
    TPR = sum(pred[defect_idx])/len(defect_idx)

    # identify ids of the image patches in the test data that are predicted
    # as defect
    ID1 = np.argwhere(pred[0:len(datasets.val_feats)] == 1)
    ID2 = np.argwhere(pred[len(datasets.val_feats):len(pred)] == 1)
    # print("shape of ID1, ID2: ", ID1.shape, ID2.shape)


    overlayresults(datasets, id_p, id_n, angl)
    return tpr1, time2




def generateROC(labels, scores, filename, angl):
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    path = os.path.dirname(os.path.realpath(__file__))
    path = path.replace("source", "data/" + str(angl) + "/roc/") 
    if not os.path.exists(path):
        os.makedirs(path)

    fpr,tpr,thresholds=roc_curve(labels, scores, pos_label=0)

    # the highest TPR when FPR = 0
    tpr0idx = max(np.argwhere(fpr == 0))
    TPR0 = tpr[tpr0idx]
    # print("TPR max: ", TPR0)


    roc_auc=auc(fpr, tpr)
    string = "ROC with TPR = {0} when FPR = 0".format(TPR0)
    plt.title(string)
    plt.plot(fpr, tpr,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.savefig(path + filename.split(".")[0] + '.png')
    plt.close()
    # plt.show()

    return fpr, tpr, thresholds, roc_auc, TPR0[0]

    
# def postprocess_latent_code_gabor_withoutAE(filename, datasets = None):

#     if datasets == None:
#         path = os.path.dirname(os.path.realpath(__file__))
#         path = path.replace("source", "data/")
#         datasets = fabricSet_One(path, filename, patch_size=patch_size, stride=4, transform = None)

#     train_p = len(datasets.train_feats)
#     val_p = len(datasets.val_feats)
#     val_n = len(datasets.neg_feats)


#     in_train_p = []
#     in_val_p = []
#     in_val_n = []


#     for i in range(train_p):
#         in_train_p.append(datasets.train_feats[i])
    

#     for i in range(val_p):
#         in_val_p.append(datasets.val_feats[i])

#     for i in range(val_n):
#         in_val_n.append(datasets.neg_feats[i])


#     code_out_train_p = np.stack(in_train_p, axis=0)
#     code_out_val_p = np.stack(in_val_p, axis=0)
#     code_out_n = np.stack(in_val_n, axis=0)

#     # print("shape of train", code_out_train_p.shape)
#     # print("shape of val", code_out_val_p.shape)
#     # print("shape of def", code_out_n.shape)

#     # train = np.concatenate(code_out_train_p,code_out_val_p)
#     # C = np.diag(np.cov(train));
#     # print("shape of C: ", C.shape)
    

#     train_dist = squareform(pdist(code_out_train_p))  # , 'mahalanobis', VI=None
#     diag = np.ones(train_dist.shape)*1.0e+10
#     train_dist += np.diag(np.diag(diag)) 
#     train_dist = np.amin(train_dist, axis=1)
#     # print("pdist of train", train_dist.shape)



#     testp_dist = np.amin(cdist(code_out_val_p, code_out_train_p), axis=1)
#     nnidp      = np.argmin(cdist(code_out_val_p, code_out_train_p), axis=1)

#     testn_dist = np.amin(cdist(code_out_n, code_out_train_p), axis=1)
#     nnidn      = np.argmin(cdist(code_out_n, code_out_train_p), axis=1)

#     # print("reshape of testp_dist", testp_dist.shape)
#     # print("reshape of nnidp", nnidp.shape)

#     # print("reshape of testn_dist", testn_dist.shape)
#     # print("reshape of nnidn", nnidn.shape)

#     labels = np.concatenate((np.ones((testp_dist.shape[0],1)),np.zeros((testn_dist.shape[0],1))), axis=0)
#     # print("label shape: ",labels.shape)

#     # for i in range(len(nnidp)):
#     #     testp_dist[i] = testp_dist[i] / train_dist[nnidp[i]]
#     # for i in range(len(nnidn)):
#     #     testn_dist[i] = testn_dist[i] / train_dist[nnidn[i]]

#     scores = np.concatenate((testp_dist ,testn_dist), axis=0)

#     # print("score shape: ",scores.shape)

#     maxratio = max(scores[0:testp_dist.shape[0]-1])
#     # print("max ratio is: ", maxratio)

#     pred = scores > maxratio
#     id_p = []
#     id_n = []

#     # print("FINDING OUTLIERS")

#     for i in range(testp_dist.shape[0]):
#         if pred[i] == True:
#             id_p.append(i)

    
#     for i in range(testp_dist.shape[0], len(pred)):
#         if pred[i] == True:
#             id_n.append(i-testp_dist.shape[0])


#     fpr, tpr, thresholds, roc_auc, tpr1 = generateROC(labels, scores, filename)


#     normal_idx = np.argwhere(labels == 1)
#     # print("normal_idx shape: ", normal_idx.shape)
#     FPR = sum(pred[normal_idx])/len(normal_idx)

#     defect_idx = np.argwhere(labels == 0)
#     # print("defect_idx shape: ", defect_idx.shape)
#     TPR = sum(pred[defect_idx])/len(defect_idx)

#     # print("FPR, TPR: ", FPR, TPR)

#     # identify ids of the image patches in the test data that are predicted
#     # as defect
#     ID1 = np.argwhere(pred[0:len(datasets.val_feats)] == 1)
#     ID2 = np.argwhere(pred[len(datasets.val_feats):len(pred)] == 1)
#     # print("shape of ID1, ID2: ", ID1.shape, ID2.shape)


#     overlayresults(datasets, id_p, id_n)
#     return tpr1