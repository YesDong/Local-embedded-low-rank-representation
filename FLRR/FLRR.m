function [ lable ] = FLRR( X,cluster ,puk,numda)
[N,D]=size(X);
Z=(numda*eye(N)+X*X')\(X*X');
S1=abs(Z)+abs(Z');
%��öȾ���D
D1 = full(sparse(1:N, 1:N, sum(S1))); %���Դ˴�DΪ���ƶȾ���S��һ��Ԫ�ؼ������ŵ��Խ����ϣ��õ��Ⱦ���D
% ���������˹���� Do laplacian, L = D^(-1/2) * S * D^(-1/2)
L1 = eye(N)-(D1^(-1/2) * S1 * D1^(-1/2)); %������˹����
% ���������� V
%  eigs 'SM';����ֵ��С����ֵ
[V1, ~] = eigs(L1, puk, 'SM');
% ������������k-means
lable=kmeans(V1,cluster);
end

