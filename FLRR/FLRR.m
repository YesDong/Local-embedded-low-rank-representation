function [ lable ] = FLRR( X,cluster ,puk,numda)
[N,D]=size(X);
Z=(numda*eye(N)+X*X')\(X*X');
S1=abs(Z)+abs(Z');
%获得度矩阵D
D1 = full(sparse(1:N, 1:N, sum(S1))); %所以此处D为相似度矩阵S中一列元素加起来放到对角线上，得到度矩阵D
% 获得拉普拉斯矩阵 Do laplacian, L = D^(-1/2) * S * D^(-1/2)
L1 = eye(N)-(D1^(-1/2) * S1 * D1^(-1/2)); %拉普拉斯矩阵
% 求特征向量 V
%  eigs 'SM';绝对值最小特征值
[V1, ~] = eigs(L1, puk, 'SM');
% 对特征向量求k-means
lable=kmeans(V1,cluster);
end

