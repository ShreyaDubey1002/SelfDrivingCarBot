function [ fX ] = sigmoid( X,W,size )
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
fX=zeros(size,1);
fX=X*W;
%fX
for i=1:size
    fX(i,1)=1/(1+exp(-fX(i,1)));
%[m,n]=size(fX)
end

