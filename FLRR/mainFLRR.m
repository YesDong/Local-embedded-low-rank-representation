close all
clear all
clc

% load('H:\m_work\data\doubleswissroll1200.dat');
% X=doubleswissroll1200(:,1:3);
% L=doubleswissroll1200(:,4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D=load('H:\m_work\data\jain.txt');
% X=D(:,1:2);
% L=D(:,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% N=[2,5,7,10,13];
N=[22,29,30,34,36];
[X, gnd] = face_pick_ORL10(N);
X = X/256;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cluster=max(gnd);
puk=cluster;
Numda=0.4;
tic;
for numda=1:length(Numda)
lable=FLRR( X,cluster ,puk,Numda(numda));
toc;
[miss,index] = missclassGroups(lable,gnd,cluster);
acc(numda)=1-miss/length(gnd)
time=toc/5
end