function [X, gnd] = face_pick_10(i,j)

load('H:\m_work\data\YaleBCrop025.mat');
% ind = j*(i-1)+1:j*i;
ind=i;
X = Y(:, :, ind);
X = reshape(X, [size(Y, 1), 64*j]);
gnd = 1:j;
gnd =repmat(gnd, [64 1]);
gnd = gnd(:);