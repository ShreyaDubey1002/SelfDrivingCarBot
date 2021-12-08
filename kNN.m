clc;
clear;
data = importdata('notMNIST_small.mat');
images = data.images;
labels = data.labels;
[a,b,c] = size(images);
X = zeros(c,a*b);
l=1;
for i=1:c
    for j=1:a
        for k=1:b
            X(i,l) = images(j,k,i);
            l = l+1;
        end
    end
    l=1;
end

%splitting into training and validation
iporder = randperm(c);
ratio = 0.99;

n1 = fix(ratio*c);
n2 = c - n1;

n2
trainX = zeros(n1,a*b);
trainY = zeros(n1,1);
validationX = zeros(n2,a*b);
validationY = zeros(n2,1);
for i=1:n1
    trainX(i,:) = X(iporder(i),:);
    trainY(i,1) = labels(iporder(i),1);
end

for i=1:n2
    validationX(i,:) = X(iporder(n1+i),:);
    validationY(i,1) = labels(iporder(n1+i),1);
end

%kNN
k = 5;
out = zeros(n2,1);
for i=1:n2
    i
    tempDistance = zeros(n1,1);
    for j=1:n1
        sum=0;
        for q=1:a*b
            sum = sum + (validationX(i,q)-trainX(j,q)).^2;
        end
        tempDistance(j,1) = sum;
    end
    ind = zeros(k,1);
    for q=1:k
        [val,ind(q)] = min(tempDistance(:,1));
        ind(q)
        tempDistance(ind(q)) = []; 
    end
    out(i) = mode(ind);
end


%error calculation
error = 0;
for i=1:n2
    if(out(i)~=validationY(i))
        error = error+1;
    end
end

error = error*100/c