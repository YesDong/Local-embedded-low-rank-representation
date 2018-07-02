function [X, gnd] = face_pick_ORL10(N)
load('H:\m_work\data\att_faces\ORL4656matrix.mat');
X=[];
for i=1:length(N)
Z=S2((i-1)*10+1:i*10,:);
X=[X;Z];
end
gnd = 1:length(N);
gnd =repmat(gnd, [10 1]);
gnd = gnd(:);