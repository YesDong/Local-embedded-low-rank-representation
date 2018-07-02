# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:39:17 2018

@author: yesds
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize 
def LLE(X,kn,d):
    """doc."""
    (datasize,dataattribute)=X.shape
    dis=np.ones([datasize,datasize])
    W=np.zeros([datasize,datasize])
    sortdis=np.zeros([datasize,kn],dtype=np.int)
    for i in range(datasize):
        diff=np.tile(X[i],(datasize,1))-X
        diff2=diff**2
        dis[i]=diff2.sum(axis=1)
        sortdis[i]=dis[i].argsort()[range(1,kn+1)]
        print(sortdis[i])
#        
        Z=np.dot(diff[sortdis[i]],np.transpose(diff[sortdis[i]]))+np.eye(kn)
        Zin=LA.inv(Z)
        W[i][sortdis[i]]=np.dot(np.ones(kn),np.transpose(Zin)) \
        /np.dot(np.dot(np.ones(kn),Zin),np.ones([kn,1]))

    L=np.dot(np.eye(datasize)-np.transpose(W),np.eye(datasize)-W)
    
    eaL=np.max(abs(L-np.transpose(L)))
    print(eaL)
    eigvals, eigvecs = eigsh(L,k=2,sigma=0.0)
 
    # 取出前k小的特征值对应的特征向量，并进行正则化  
    
    return eigvecs

###########zikongjian
def FLRR(X, numda,k):
    XS=X.shape
    Z=np.dot(LA.inv( numda *np.eye(XS[1]) \
+np.dot((np.transpose(X)),X)),np.dot((np.transpose(X)),X))
    Z=Z+np.transpose(Z)
    Dn = np.diag(np.power(np.sum(Z, axis=1), -2))
    
        # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn  
    # 也是做了数学变换的，简写为下面一行  
    L = np.eye(len(X[0])) - np.dot(np.dot(Dn, Z), Dn)

    eigvals, eigvecs = LA.eig(L) 

#    # 前k小的特征值对应的索引，argsort函数  
    indices = np.argsort(eigvals)[:k]
#    # 取出前k小的特征值对应的特征向量，并进行正则化  
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])  
    return [k_smallest_eigenvectors,Z]


DA = np.loadtxt('H:\py_work\data1\jain.txt')
X=DA[:,0:2]
l=DA[:,2]
Y=LLE(X,10,2)
[tz,Z]= FLRR(Y.T, 0.7, 2)
pl=KMeans(n_clusters=2).fit_predict(tz)
fig, (ax0, ax1) = plt.subplots(ncols=2)  
ax0.scatter(X[:, 0], X[:, 1], c=l)  
ax0.set_title('raw data')
#ax1.scatter(Y[:, 0], Y[:, 1], c=l)
ax1.scatter(X[:, 0], X[:, 1], c=pl)  
ax1.set_title('LLE') 
plt.show() 
fig, 
ax=plt.scatter(Y[:, 0], Y[:, 1], c=l)  
plt.title('LLE_Y')
plt.show() 
fig, 
ax=plt.scatter(tz[:, 0], tz[:, 1], c=l)  
plt.title('LLE')
plt.show() 