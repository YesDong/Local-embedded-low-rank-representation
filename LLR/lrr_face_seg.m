% function [] = lrr_face_seg()
% data = loadmatfile('yaleb10.mat');
% 
% X = data.X;
% gnd = data.cids;
% K = max(gnd);
close all
clear all
clc
accuracy = [];
runTime = 0;
% for i=1:4
% for count =1:loops
for num=1:1
tic;
%%%%%%%%%%%%%%%%%%%%%%%%
% [X, gnd] = face_pick_10(1,5);
% X = X/256;
% X=X/max(max(X));
%%%%%%%%%%%%%%%%%%%%%%%%
%doubleswissroll
% DA = load('H:\m_work\data\jain.txt');
% X=DA(:,1:2);
% gnd=DA(:,3);
% X=X'/max(max(X));
%%%%%%%%%%%%%%%%%%%%%%%%
% N=[22,29,30,34,36];
N=[2,5,7,10,13];
[X, gnd] = face_pick_ORL10(N);
X = X'/256;
%%%%%%%%%%%%%%%%%%%%%%%%%
% N=6:10;
% N=[7,6,3,5,2];
% [X, gnd] = face_pick_JAFFE10(N);
% X=X'/256;
%%%%%%%%%%%%%%%%%%%%%%%%%
gnd = gnd';
K = max(gnd');
% load('H:\m_work\data\YaleB_32x32.mat');
% da1=4;
% da2=5;
% X=[fea((da1-1)*64+1:da1*64,:);fea((da2-1)*64+1:da2*64,:)];
% X=X'/256;
% gnd=[ones(64,1);2*ones(64,1)];
% K = max(gnd');
%run lrr
Z = solve_lrr(X,0.15);
% Q = orth(X');
% A = X*Q;
% Z1 = lrra(X,A,0.2);
% Z = Q*Z1;
%post processing
[U,S,V] = svd(Z,'econ');
S = diag(S);
r = sum(S>1e-4*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^4;

% spectral clustering
D = diag(1./sqrt(sum(L,2)));
L = D*L*D;
[U,S,V] = svd(L);
V = U(:,1:K);
V = D*V;

n = size(V,1);
% M = zeros(K,K,20);
% rand('state',123456789);
% for i=1:size(M,3)
%     inds = false(n,1);
%     while sum(inds)<K
%         j = ceil(rand()*n);
%         inds(j) = true;
%     end
%     M(:,:,i) = V(inds,:);
% end

runTime = runTime + toc;
idx = kmeans(V,K,'emptyaction','singleton');
% idx = kmeans(V,K,'emptyaction','singleton','start',M,'display','off');
acc =  1 - missclassGroups(idx,gnd,K)/length(idx);

runTime = runTime + toc;

accuracy(num)=  acc;
% end
% runTime = runTime/loops
aveage_acc = mean(accuracy);
disp(['seg acc=' num2str(aveage_acc)]);
toc;
time(num)=toc;
meantime=sum(time)/20;
end
% disp(['accuracy std=' num2str(std(accuracy))]);
% normZ = Z - min(Z(:));
% normZ = normZ ./ max(normZ(:)); % *
% 
% imshow(normZ)