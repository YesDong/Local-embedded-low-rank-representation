# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:10:25 2018

@author: yesds
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.datasets import make_blobs 
from sklearn.cluster import KMeans
#from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize 
###

#numda=0.5
def FLRR(X, numda,k):
    XS=X.shape
    Z=np.dot(LA.inv( numda *np.eye(XS[1]) \
+np.dot((np.transpose(X)),X)),np.dot((np.transpose(X)),X))
    Z=Z+np.transpose(Z)
    Dn = np.diag(np.power(np.sum(Z, axis=1), -0.5))
    
        # 拉普拉斯矩阵：L=Dn*(D-W)*Dn=I-Dn*W*Dn  
    # 也是做了数学变换的，简写为下面一行  
    L = np.eye(len(X[0])) - np.dot(np.dot(Dn, Z), Dn)

    eigvals, eigvecs = LA.eig(L) 

#    # 前k小的特征值对应的索引，argsort函数  
    indices = np.argsort(eigvals)[:k]
#    # 取出前k小的特征值对应的特征向量，并进行正则化  
    k_smallest_eigenvectors = normalize(eigvecs[:, indices])  
    return [k_smallest_eigenvectors,Z]
       

#X ,l= make_blobs(n_features=9, centers=2)
DA = np.loadtxt('H:\py_work\data1\jain.txt')
X=DA[:,0:2]
l=DA[:,2]
X=np.transpose(X)
[tz,Z]= FLRR(X, 0.7, 2)

# 利用KMeans进行聚类  
pl=KMeans(n_clusters=2).fit_predict(tz)
# 画图  
plt.style.use('ggplot')  
# 原数据  
fig, ax = plt.subplots(nrows=1,ncols=2)  
ax[0].scatter(X[0, :], X[1, :], c=l)  
ax[0].set_title(' raw data') 
ax[1].scatter(X[0, :], X[1, :], c=pl)  
ax[1].set_title(' FLRR raw data')
#fig.suptitle('raw data') 
#f1.set_title('raw data')
# 谱聚类结果  
fig, ( [ax2,ax3]) = plt.subplots(nrows=1,ncols=2)  
ax2.scatter(tz[:, 0], tz[:, 1], c=l)  
ax2.set_title('eigenvector_raw_lable') 
ax3.scatter(tz[:, 0], tz[:, 1], c=pl)  
ax3.set_title('eigenvector_lable') 
plt.show()  