data = importdata('notMNIST_small.mat');
images = data.images;
labels = data.labels;
[a,b,c] = size(images);
X = zeros(c,a*b+1);
l=2;
X(:,1) = 1;
for i=1:c
    for j=1:a
        for k=1:b
            X(i,l) = images(j,k,i); 
            l = l+1;
        end
    end
    l=2;
end


%gradient descent

W=zeros(a*b+1,1);
W(1,1)=0.04;

alpha=0.01;
for j=1:c
    W=W-(alpha*(transpose(X)*(sigmoid(X,W,c)-labels)));
    fX=sigmoid(X,W,c);
    total=0;
    fX=round(fX);
    for i=1:c 
        if fX(i,1)==labels(i,1)
            total=total+1;
        end
    end
    j
    accuracy=total;
end
accuracy
