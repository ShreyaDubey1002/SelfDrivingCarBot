
% This function is the primary driver for homework 3 part 1
function l3a
close all;
clear all;
clc;

sd=0.2;

% number of data points per class


rand('seed', 1);
N = 21999;
D = 1024;

X = double(importdata('X.mat'));
Y = double(importdata('Y.mat'));

testX = double(importdata('testX.mat'));

N=size(X,1);
D=size(X,2);

% number of epochs for training
nEpochs = 1000;

% learning rate
eta = 0.01;
% number of hidden layer units
H1 = 400;
H2 = 70;
H3 = 40;
% train the MLP using the generated sample dataset
[w, v1,v2, trainerror] = mlptrain(X,Y, H1,H2,H3, eta, nEpochs,testX);

function [w, v1,v2, trainerror] = mlptrain(X, Y, H1,H2,H3, eta, nEpochs,testX)
%cols=1024
% X - training dtrainingata of size NxD
% Y -  labels of size NxK
% H - the number of hidden layers
% eta - the learning rate
% nEpochs - the number of training epochs
% define and initialize the neural network parameters

% number of training data points
N = size(X,1);
% N=21999
% number of inputs
D = size(X,2); % excluding the bias term
% D=1025
% number of outputs
K = size(Y,2);
% K=1
batchSize = 32;

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% w is a HxD matrix
w = -0.01+(0.02)*rand(H1,D);

% weights for the connections between input and hidden layer
% random values from the interval [-0.3 0.3]
% v is a Kx(H+1) matrix
v1 = -0.01+(0.02)*rand(H2,(H1+1));

v2 = -0.01+(0.02)*rand(H3,(H2+1));

v3 = -0.01+(0.02)*rand(K,(H3+1));

dropoutPercent = 0;
% randomize the order in which the input data points are presented to the
% MLP

n1 = fix(0.8*N);
n1=17664;
n2 = N-n1;
n3 = size(testX,1);
% n1 is for training 
% n2 is for validation
trainX = zeros(n1,D);
trainY = zeros(n1,1);

validationX = zeros(n2,D);
validationY = zeros(n2,1);
iporder = randperm(N);
for i=1:n1
    trainX(i,1:D) = X(iporder(i),:);
    trainY(i,:) = Y(iporder(i),:);
end
for i=1:n2
    validationX(i,1:D) = X(iporder(i+n1),:);
    validationY(i,:) = Y(iporder(i+n1),:);
end
deltav3=zeros(1,H3+1);
deltav2 = zeros(H3,H2+1);
deltav1 = zeros(H2,H1+1);
deltaw = zeros(H1,D);

% mlp training through stochastic gradient descent
numberofBatches=n1/batchSize;

for epoch = 1:nEpochs
    start=1;
    iporderInput=randperm(D);
    iporderH1=randperm(H1+1);
    iporderH2=randperm(H2+1);
    iporderH3=randperm(H3+1);
      
    for n = 1:numberofBatches
        %dropout
        trainXnew =zeros(batchSize,D);
        trainXnew = trainX(start:start+batchSize-1,:);
        trainYnew=zeros(batchSize,K);
        trainYnew = trainY(start:start+batchSize-1,:);
        

        for i=1:fix(dropoutPercent*(D))
            trainXnew(:,iporderInput(i))=0; 
        end
        % forward pass
        %h1=batchSize x H1
        h1 = trainXnew*transpose(w);
        h1=sigmf(h1,[1 0]);
        h1new = zeros(batchSize,H1+1);
        h1new(:,1) = 1;
        h1new(:,2:H1+1) = h1(:,:);
       
        
        for i=1:fix(dropoutPercent*(H1+1))
            h1new(:,iporderH1(i))=0;
        end
        
        
        %h2= batchSize x H2
        h2 = h1new*transpose(v1);
        
        h2=sigmf(h2,[1 0]);
       
        h2new = zeros(batchSize,H2+1);
        h2new(:,1) = 1;
        h2new(:,2:H2+1) = h2(:,:);
        
        
        for i=1:fix(dropoutPercent*(H2+1))
            h2new(:,iporderH2(i))=0;
        end
        
         h3 = h2new*transpose(v2);
        
        h3=sigmf(h3,[1 0]);
       
        h3new = zeros(batchSize,H3+1);
        h3new(:,1) = 1;
        h3new(:,2:H3+1) = h3(:,:);
        
        
        for i=1:fix(dropoutPercent*(H3+1))
            h3new(:,iporderH3(i))=0;
        end
       
        %o= batchSize x K
        o=zeros(batchSize,K);
        o = h3new*transpose(v3);
        
       
        %minibatch method
        % delta3 = batchSize x K
            delta3=o-trainYnew;   % batchsize x K
            
            delta2=(delta3*v3).*(h3new).*(1-h3new);  %  batchsize x (H3+1)
            delta2=delta2(:,2:size(delta2,2));    %  batchsize x (H3)
            
            delta1=(delta2*v2).*h2new.*(1-h2new);
            delta1=delta1(:,2:size(delta1,2));  % batchsize x H2
            
            delta0=(delta1*v1).*h1new.*(1-h1new);
            delta0=delta0(:,2:size(delta0,2)); % batchsize X H1
            
            deltav3 = eta*delta3'*h3new;
            deltav2 = eta*delta2'*h2new;
            deltav1 = eta*delta1'*h1new;
            deltaw = eta*delta0'*trainXnew;
        
            w=w-(deltaw/batchSize);
            v1=v1-(deltav1/batchSize);
            v2=v2-(deltav2/batchSize);
            v3=v3-(deltav3/batchSize);
          
      start=start+batchSize;  
    end
    
    
    ydash = mlptest(trainX, w, v1,v2,v3,dropoutPercent);
    % compute the training error
    % ---------
    %'TO DO'% uncomment the next line after adding the necessary code
    trainerror(epoch) = 0;
    for i=1:n1
            trainY(i,1);
            ydash(i,1);
            trainerror(epoch) = trainerror(epoch) + 0.5*((trainY(i,1)-ydash(i,1)).^2);
    end
    trainerror(epoch) = trainerror(epoch)/n1;
    % ---------
    disp(sprintf('training error after epoch %d: %f\n',epoch,...
        trainerror(epoch)));
      
    %calculating validation error
    ydash2 = mlptest(validationX, w, v1,v2,v3,dropoutPercent);
    
    validationError(epoch) = 0;
    for i=1:n2
            validationY(i,1);
            ydash2(i,1);
            validationError(epoch) = validationError(epoch) + 0.5*((validationY(i,1)-ydash2(i,1)).^2);
    end
    validationError(epoch) = validationError(epoch)/n2;
    
    % ---------
    disp(sprintf('Validation error after epoch %d: %f\n',epoch,...
        validationError(epoch)));
   
    
    %writing to output file
    if(epoch == nEpochs)
        ydash3 = mlptest(testX, w, v1,v2,v3,dropoutPercent);
        fileID = fopen('test-data-output.txt','w');
        for i=1:n3
            fprintf(fileID,'%s %f\n',strcat('./test/img_', int2str(i-1), '.jpg'),ydash3(i,1));
        end
    end
    
    
end

Xlinspace = 1:1:nEpochs;
plot(Xlinspace,trainerror,'linewidth',2,'color','k')
hold on;
plot(Xlinspace,validationError,'linewidth',2,'color','r')
legend('Train error', 'Validation error', 'Location', 'NorthOutside', ...
    'Orientation', 'horizontal');

return;

function ydash = mlptest(X, w, v1,v2,v3,dropoutPercent)
% forward pass of the network
N = size(X,1);
D = size(X,2);
H1 = size(w,1);
H2 = size(v1,1);
H3 = size(v2,1);
K = size(v3,1);

h1 = (1-dropoutPercent)*X*transpose(w);
h1=sigmf(h1,[1 0]);


h1new = zeros(N,H1+1);
h1new(:,1) = 1;
h1new(:,2:H1+1) = h1(:,:);

h2 = (1-dropoutPercent)*h1new*transpose(v1);
h2=sigmf(h2,[1 0]);

h2new = zeros(N,H2+1);
h2new(:,1) = 1;
h2new(:,2:H2+1) = h2(:,:);

h3 = (1-dropoutPercent)*h2new*transpose(v2);
h3=sigmf(h3,[1 0]);

h3new = zeros(N,H3+1);
h3new(:,1) = 1;
h3new(:,2:H3+1) = h3(:,:);

o = (1-dropoutPercent)*h3new*transpose(v3);

ydash = o;

return;






















% % This function is the primary driver for homework 3 part 1
% function l3a
% close all;
% clear all;
% clc;
% 
% sd=0.2;
% 
% % number of data points per class
% 
% 
% rand('seed', 1);
% N = 21999;
% D = 1024;
% 
% X = double(importdata('X.mat'));
% Y = double(importdata('Y.mat'));
% 
% N=size(X,1);
% D=size(X,2);
% %X=double(X);
% %X=normc(X);
% % X = X - min(X(:));
% % X = X ./ max(X(:)); 
% 
% % no of rows in X=21999
% % no of cols in X=1025
% 
% % number of epochs for training
% nEpochs = 1000;
% 
% % learning rate
% eta = 0.01;
% numHidden = 2;
% H1 = 512;
% H2 = 64;
% 
% % number of hidden layer units
% 
%     
%     % train the MLP using the generated sample dataset
%     [w, v1,v2, trainerror] = mlptrain(X,Y, H1,H2, eta, nEpochs);
% 
% function [w, v1,v2, trainerror] = mlptrain(X, Y, H1,H2, eta, nEpochs)
% %cols=1024
% % X - training dtrainingata of size NxD
% % Y -  labels of size NxK
% % H - the number of hidden layers
% % eta - the learning rate
% % nEpochs - the number of training epochs
% % define and initialize the neural network parameters
% 
% % number of training data points
% N = size(X,1);
% % N=21999
% % number of inputs
% D = size(X,2); % excluding the bias term
% % D=1025
% % number of outputs
% K = size(Y,2);
% % K=1
% batchSize = 32;
% 
% % weights for the connections between input and hidden layer
% % random values from the interval [-0.3 0.3]
% % w is a HxD matrix
% w = -0.01+(0.02)*rand(H1,D);
% 
% % weights for the connections between input and hidden layer
% % random values from the interval [-0.3 0.3]
% % v is a Kx(H+1) matrix
% v1 = -0.01+(0.02)*rand(H2,(H1+1));
% 
% v2 = -0.01+(0.02)*rand(K,(H2+1));
% dropoutPercent = 0.5;
% % randomize the order in which the input data points are presented to the
% % MLP
% 
% % x = zeros(N,D);
% % x(:,1) = 1;
% % x(:,2:D+1) = X(:,:);
% n1 = fix(0.8*N);
% n1=17664;
% n2 = N-n1;
% % n1 is for training 
% % n2 is for validation
% trainX = zeros(n1,D);
% trainY = zeros(n1,1);
% 
% % testX : validation set
% testX = zeros(n2,D);
% % testY : validation set
% testY = zeros(n2,1);
% iporder = randperm(N);
% for i=1:n1
%     trainX(i,1:D) = X(iporder(i),:);
%     trainY(i,:) = Y(iporder(i),:);
% end
% for i=1:n2
%     testX(i,1:D) = X(iporder(i+n1),:);
%     testY(i,:) = Y(iporder(i+n1),:);
% end
% %ydash = mlptest(trainX, w, v1,v2);
%             deltav2 = zeros(1,H2+1);
%             deltav1 = zeros(H2,H1+1);
%             deltaw = zeros(H1,D+1);
%            
%            
% % mlp training through stochastic gradient descent
% numberofBatches=n1/batchSize;
% 
% for epoch = 1:nEpochs
%     error = 0;
%     start=1;
%     for n = 1:numberofBatches
%      
%         %dropout
%         
%         
%         trainXnew =zeros(batchSize,D);
%         trainXnew = trainX(start:start+batchSize-1,:);
%         
%         trainYnew=zeros(batchSize,K);
%         trainYnew = trainY(start:start+batchSize-1,:);
%         
%         
%         
%             dropoutExamples = randperm(D);
% 
%             for i=1:fix(dropoutPercent*(D))
%             trainXnew(:,dropoutExamples(i))=0; 
%             end
%             
%        
%         % forward pass
%        
%         
%         %h1=batchSize x H1
%         h11 = trainXnew*transpose(w);
%         
%         
%           % h1=arrayfun(@sigmoid,h1);
%           h1=sigmf(h11,[1 0]);
%            
%         
%         % ---------
%         % hidden to output layer
%         % calculate the output of the output layer units - ydash
%         % ---------
%         %'TO DO'%
%         
%         h1new = zeros(batchSize,H1+1);
%         h1new(:,1) = 1;
%         h1new(:,2:H1+1) = h1(:,:);
%        
%         dropoutExamplesh1 = randperm(H1+1);
%         for i=1:fix(dropoutPercent*(H1+1))
%             h1new(:,dropoutExamplesh1(i))=0;
%         end
%         
%         
%         %h2= batchSize x H2
%         h2 = h1new*transpose(v1);
%         
%         %h2=arrayfun(@sigmoid,h2);
%         
%         h2=sigmf(h2,[1 0]);
%        
%         h2new = zeros(batchSize,H2+1);
%         h2new(:,1) = 1;
%         h2new(:,2:H2+1) = h2(:,:);
%         
%         dropoutExamplesh2 = randperm(H2+1);
%         for i=1:fix(dropoutPercent*(H2+1))
%             h2new(:,dropoutExamplesh2(i))=0;
%             
%         end
%        
%         %o= batchSize x K
%         o=zeros(batchSize,K);
%         o = h2new*transpose(v2);
%        
%          if n==1
%         output= o(1,1);
%     end
%        
%         
%         % ---------
%         
%         % backward pass
% 
%             %minibatch method
%         % delta3 = batchSize x K
%         
%         
%             delta3=o-trainYnew;   % batchsize x K
%             
%             delta2=(delta3*v2).*(h2new).*(1-h2new);  %  batchsize x (H2+1)
%             delta2=delta2(:,2:size(delta2,2));    %  batchsize x (H2)
%             
%             delta1=(delta2*v1).*h1new.*(1-h1new);
%             delta1=delta1(:,2:size(delta1,2));  % batchsize x H1
%             
%             deltav2= eta*delta3'*h2new;
%             deltav1= eta*delta2'*h1new;
%             deltaw= eta*delta1'*trainXnew;
%         
%         
% %    delta3=o-trainYnew;
% %         
% %    delta2=(delta3*v2).*(h2new.*(1-h2new));
% %    
% %  
% %     % delta2 = batchSize x H2
% %    delta2=delta2(:,2:size(delta2,2));
% %    
% %    delta1=(delta2*v1).*(h1new.*(1-h1new));
% %             delta1=delta1(:,2:size(delta1,2));  % batchsize x H1
% %    
% %    
% %    
% %             
% %             deltav2= transpose(delta3)*h2new;
% %             deltav1= transpose(delta2)*h1new;
% %             deltaw= transpose(delta1)*trainXnew;
% 
%             w=w-(deltaw/batchSize);
%             v1=v1-(deltav1/batchSize);
%             v2=v2-(deltav2/batchSize);
%           
%       start=start+batchSize;  
%     end
%     
%     
%     ydash = mlptest(trainX, w, v1,v2,dropoutPercent);
%     % compute the training error
%     % ---------
%     %'TO DO'% uncomment the next line after adding the necessary code
%     trainerror(epoch) = 0;
%     for i=1:n1
%             trainY(i,1);
%             ydash(i,1);
%             trainerror(epoch) = trainerror(epoch) + 0.5*((trainY(i,1)-ydash(i,1)).^2);
%     end
%     trainerror(epoch) = trainerror(epoch)/n1;
%     % ---------
%     disp(sprintf('training error after epoch %d: %f\n',epoch,...
%         trainerror(epoch)));
%     
%     
%     ydash2 = mlptest(testX, w, v1,v2,dropoutPercent);
%     
%     testerror(epoch) = 0;
%     for i=1:n2
%             testY(i,1);
%             ydash2(i,1);
%             testerror(epoch) = testerror(epoch) + 0.5*((testY(i,1)-ydash2(i,1)).^2);
%     end
%     testerror(epoch) = testerror(epoch)/n2;
%     % ---------
%     disp(sprintf('Validation error after epoch %d: %f\n',epoch,...
%         testerror(epoch)));
% end
% 
% Xlinspace = 1:1:nEpochs;
% plot(Xlinspace,trainerror,'linewidth',2,'color','k')
% hold on;
% plot(Xlinspace,testerror,'linewidth',2,'color','r')
% legend('Train error', 'Validation error', 'Location', 'NorthOutside', ...
%     'Orientation', 'horizontal');
% 
% return;
% 
% function ydash = mlptest(X, w, v1,v2,dropoutPercent)
% % forward pass of the network
% N = size(X,1);
% D = size(X,2);
% H1 = size(w,1);
% H2 = size(v1,1);
% % number of inputs
% 
% 
% % number of outputs
% K = size(v2,1);
% 
% % forward pass to estimate the outputs
% % --------------------------------------
% % input to hidden for all the data points
% % calculate the output of the hidden layer units
% % ---------
% %'TO DO'%
% h1 = (1-dropoutPercent)*X*transpose(w);
% 
% 
%    % h1=arrayfun(@sigmoid,h1);
%     h1=sigmf(h1,[1 0]);
% 
% 
% h1new = zeros(N,H1+1);
% h1new(:,1) = 1;
% h1new(:,2:H1+1) = h1(:,:);
% 
% h2 = (1-dropoutPercent)*h1new*transpose(v1);
% 
%    % h2=arrayfun(@sigmoid,h2);
%     
%     h2=sigmf(h2,[1 0]);
% 
% h2new = zeros(N,H2+1);
% h2new(:,1) = 1;
% h2new(:,2:H2+1) = h2(:,:);
% 
% o = (1-dropoutPercent)*h2new*transpose(v2);
% 
% ydash = o;
% 
% return;










% 
% % This function is the primary driver for homework 3 part 1
% function l3a
% close all;
% clear all;
% clc;
% 
% sd=0.2;
% 
% % number of data points per class
% 
% 
% rand('seed', 1);
% N = 21999;
% D = 1024;
% 
% X = double(importdata('X.mat'));
% Y = double(importdata('Y.mat'));
% 
% testX = double(importdata('testX.mat'));
% 
% N=size(X,1);
% D=size(X,2);
% 
% % number of epochs for training
% nEpochs = 1000;
% 
% % learning rate
% eta = 0.001;
% % number of hidden layer units
% H1 = 512;
% H2 = 64;
% % train the MLP using the generated sample dataset
% [w, v1,v2, trainerror] = mlptrain(X,Y, H1,H2, eta, nEpochs,testX);
% 
% function [w, v1,v2, trainerror] = mlptrain(X, Y, H1,H2, eta, nEpochs,testX)
% %cols=1024
% % X - training dtrainingata of size NxD
% % Y -  labels of size NxK
% % H - the number of hidden layers
% % eta - the learning rate
% % nEpochs - the number of training epochs
% % define and initialize the neural network parameters
% 
% % number of training data points
% N = size(X,1);
% % N=21999
% % number of inputs
% D = size(X,2); % excluding the bias term
% % D=1025
% % number of outputs
% K = size(Y,2);
% % K=1
% batchSize = 32;
% 
% % weights for the connections between input and hidden layer
% % random values from the interval [-0.3 0.3]
% % w is a HxD matrix
% w = -0.01+(0.02)*rand(H1,D);
% 
% % weights for the connections between input and hidden layer
% % random values from the interval [-0.3 0.3]
% % v is a Kx(H+1) matrix
% v1 = -0.01+(0.02)*rand(H2,(H1+1));
% 
% v2 = -0.01+(0.02)*rand(K,(H2+1));
% dropoutPercent = 0.5;
% % randomize the order in which the input data points are presented to the
% % MLP
% 
% n1 = fix(0.8*N);
% n1=17664;
% n2 = N-n1;
% n3 = size(testX,1);
% % n1 is for training 
% % n2 is for validation
% trainX = zeros(n1,D);
% trainY = zeros(n1,1);
% 
% validationX = zeros(n2,D);
% validationY = zeros(n2,1);
% iporder = randperm(N);
% for i=1:n1
%     trainX(i,1:D) = X(iporder(i),:);
%     trainY(i,:) = Y(iporder(i),:);
% end
% for i=1:n2
%     validationX(i,1:D) = X(iporder(i+n1),:);
%     validationY(i,:) = Y(iporder(i+n1),:);
% end
% deltav2 = zeros(1,H2+1);
% deltav1 = zeros(H2,H1+1);
% deltaw = zeros(H1,D+1);
% 
% % mlp training through stochastic gradient descent
% numberofBatches=n1/batchSize;
% 
% for epoch = 1:nEpochs
%     start=1;
%     iporderInput=randperm(D);
%     iporderH1=randperm(H1+1);
%     iporderH2=randperm(H2+1);
%       
%     for n = 1:numberofBatches
%         %dropout
%         trainXnew =zeros(batchSize,D);
%         trainXnew = trainX(start:start+batchSize-1,:);
%         trainYnew=zeros(batchSize,K);
%         trainYnew = trainY(start:start+batchSize-1,:);
%         
% 
%         for i=1:fix(dropoutPercent*(D))
%             trainXnew(:,iporderInput(i))=0; 
%         end
%         % forward pass
%         %h1=batchSize x H1
%         h11 = trainXnew*transpose(w);
%         h1=sigmf(h11,[1 0]);
%         h1new = zeros(batchSize,H1+1);
%         h1new(:,1) = 1;
%         h1new(:,2:H1+1) = h1(:,:);
%        
%         
%         for i=1:fix(dropoutPercent*(H1+1))
%             h1new(:,iporderH1(i))=0;
%         end
%         
%         
%         %h2= batchSize x H2
%         h2 = h1new*transpose(v1);
%         
%         h2=sigmf(h2,[1 0]);
%        
%         h2new = zeros(batchSize,H2+1);
%         h2new(:,1) = 1;
%         h2new(:,2:H2+1) = h2(:,:);
%         
%         
%         for i=1:fix(dropoutPercent*(H2+1))
%             h2new(:,iporderH2(i))=0;
%         end
%        
%         %o= batchSize x K
%         o=zeros(batchSize,K);
%         o = h2new*transpose(v2);
%         
%        
%         %minibatch method
%         % delta3 = batchSize x K
%             delta3=o-trainYnew;   % batchsize x K
%             
%             delta2=(delta3*v2).*(h2new).*(1-h2new);  %  batchsize x (H2+1)
%             delta2=delta2(:,2:size(delta2,2));    %  batchsize x (H2)
%             
%             delta1=(delta2*v1).*h1new.*(1-h1new);
%             delta1=delta1(:,2:size(delta1,2));  % batchsize x H1
%             
%             deltav2= eta*delta3'*h2new;
%             deltav1= eta*delta2'*h1new;
%             deltaw= eta*delta1'*trainXnew;
%         
%             w=w-(deltaw/batchSize);
%             v1=v1-(deltav1/batchSize);
%             v2=v2-(deltav2/batchSize);
%           
%       start=start+batchSize;  
%     end
%     
%     
%     ydash = mlptest(trainX, w, v1,v2,dropoutPercent);
%     % compute the training error
%     % ---------
%     %'TO DO'% uncomment the next line after adding the necessary code
%     trainerror(epoch) = 0;
%     for i=1:n1
%             trainY(i,1);
%             ydash(i,1);
%             trainerror(epoch) = trainerror(epoch) + 0.5*((trainY(i,1)-ydash(i,1)).^2);
%     end
%     trainerror(epoch) = trainerror(epoch)/n1;
%     % ---------
%     disp(sprintf('training error after epoch %d: %f\n',epoch,...
%         trainerror(epoch)));
%       
%     %calculating validation error
%     ydash2 = mlptest(validationX, w, v1,v2,dropoutPercent);
%     
%     validationError(epoch) = 0;
%     for i=1:n2
%             validationY(i,1);
%             ydash2(i,1);
%             validationError(epoch) = validationError(epoch) + 0.5*((validationY(i,1)-ydash2(i,1)).^2);
%     end
%     validationError(epoch) = validationError(epoch)/n2;
%     % ---------
%     disp(sprintf('Validation error after epoch %d: %f\n',epoch,...
%         validationError(epoch)));
%    
%     
%     %writing to output file
%     if(epoch == nEpochs)
%         ydash3 = mlptest(testX, w, v1,v2,dropoutPercent);
%         fileID = fopen('test-data-output.txt','w');
%         for i=1:n3
%             fprintf(fileID,'%*s %f','./test/img_' + i-1 + '.jpg',ydash3(i,1));
%         end
%     end
% end
% 
% Xlinspace = 1:1:nEpochs;
% plot(Xlinspace,trainerror,'linewidth',2,'color','k')
% hold on;
% plot(Xlinspace,validationError,'linewidth',2,'color','r')
% legend('Train error', 'Validation error', 'Location', 'NorthOutside', ...
%     'Orientation', 'horizontal');
% 
% return;
% 
% function ydash = mlptest(X, w, v1,v2,dropoutPercent)
% % forward pass of the network
% N = size(X,1);
% D = size(X,2);
% H1 = size(w,1);
% H2 = size(v1,1);
% 
% K = size(v2,1);
% 
% h1 = (1-dropoutPercent)*X*transpose(w);
% h1=sigmf(h1,[1 0]);
% 
% 
% h1new = zeros(N,H1+1);
% h1new(:,1) = 1;
% h1new(:,2:H1+1) = h1(:,:);
% 
% h2 = (1-dropoutPercent)*h1new*transpose(v1);
% h2=sigmf(h2,[1 0]);
% 
% h2new = zeros(N,H2+1);
% h2new(:,1) = 1;
% h2new(:,2:H2+1) = h2(:,:);
% 
% o = (1-dropoutPercent)*h2new*transpose(v2);
% 
% ydash = o;
% 
% return;
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % % This function is the primary driver for homework 3 part 1
% % function l3a
% % close all;
% % clear all;
% % clc;
% % 
% % sd=0.2;
% % 
% % % number of data points per class
% % 
% % 
% % rand('seed', 1);
% % N = 21999;
% % D = 1024;
% % 
% % X = double(importdata('X.mat'));
% % Y = double(importdata('Y.mat'));
% % 
% % N=size(X,1);
% % D=size(X,2);
% % %X=double(X);
% % %X=normc(X);
% % % X = X - min(X(:));
% % % X = X ./ max(X(:)); 
% % 
% % % no of rows in X=21999
% % % no of cols in X=1025
% % 
% % % number of epochs for training
% % nEpochs = 1000;
% % 
% % % learning rate
% % eta = 0.01;
% % numHidden = 2;
% % H1 = 512;
% % H2 = 64;
% % 
% % % number of hidden layer units
% % 
% %     
% %     % train the MLP using the generated sample dataset
% %     [w, v1,v2, trainerror] = mlptrain(X,Y, H1,H2, eta, nEpochs);
% % 
% % function [w, v1,v2, trainerror] = mlptrain(X, Y, H1,H2, eta, nEpochs)
% % %cols=1024
% % % X - training dtrainingata of size NxD
% % % Y -  labels of size NxK
% % % H - the number of hidden layers
% % % eta - the learning rate
% % % nEpochs - the number of training epochs
% % % define and initialize the neural network parameters
% % 
% % % number of training data points
% % N = size(X,1);
% % % N=21999
% % % number of inputs
% % D = size(X,2); % excluding the bias term
% % % D=1025
% % % number of outputs
% % K = size(Y,2);
% % % K=1
% % batchSize = 32;
% % 
% % % weights for the connections between input and hidden layer
% % % random values from the interval [-0.3 0.3]
% % % w is a HxD matrix
% % w = -0.01+(0.02)*rand(H1,D);
% % 
% % % weights for the connections between input and hidden layer
% % % random values from the interval [-0.3 0.3]
% % % v is a Kx(H+1) matrix
% % v1 = -0.01+(0.02)*rand(H2,(H1+1));
% % 
% % v2 = -0.01+(0.02)*rand(K,(H2+1));
% % dropoutPercent = 0.5;
% % % randomize the order in which the input data points are presented to the
% % % MLP
% % 
% % % x = zeros(N,D);
% % % x(:,1) = 1;
% % % x(:,2:D+1) = X(:,:);
% % n1 = fix(0.8*N);
% % n1=17664;
% % n2 = N-n1;
% % % n1 is for training 
% % % n2 is for validation
% % trainX = zeros(n1,D);
% % trainY = zeros(n1,1);
% % 
% % % testX : validation set
% % testX = zeros(n2,D);
% % % testY : validation set
% % testY = zeros(n2,1);
% % iporder = randperm(N);
% % for i=1:n1
% %     trainX(i,1:D) = X(iporder(i),:);
% %     trainY(i,:) = Y(iporder(i),:);
% % end
% % for i=1:n2
% %     testX(i,1:D) = X(iporder(i+n1),:);
% %     testY(i,:) = Y(iporder(i+n1),:);
% % end
% % %ydash = mlptest(trainX, w, v1,v2);
% %             deltav2 = zeros(1,H2+1);
% %             deltav1 = zeros(H2,H1+1);
% %             deltaw = zeros(H1,D+1);
% %            
% %            
% % % mlp training through stochastic gradient descent
% % numberofBatches=n1/batchSize;
% % 
% % for epoch = 1:nEpochs
% %     error = 0;
% %     start=1;
% %     for n = 1:numberofBatches
% %      
% %         %dropout
% %         
% %         
% %         trainXnew =zeros(batchSize,D);
% %         trainXnew = trainX(start:start+batchSize-1,:);
% %         
% %         trainYnew=zeros(batchSize,K);
% %         trainYnew = trainY(start:start+batchSize-1,:);
% %         
% %         
% %         
% %             dropoutExamples = randperm(D);
% % 
% %             for i=1:fix(dropoutPercent*(D))
% %             trainXnew(:,dropoutExamples(i))=0; 
% %             end
% %             
% %        
% %         % forward pass
% %        
% %         
% %         %h1=batchSize x H1
% %         h11 = trainXnew*transpose(w);
% %         
% %         
% %           % h1=arrayfun(@sigmoid,h1);
% %           h1=sigmf(h11,[1 0]);
% %            
% %         
% %         % ---------
% %         % hidden to output layer
% %         % calculate the output of the output layer units - ydash
% %         % ---------
% %         %'TO DO'%
% %         
% %         h1new = zeros(batchSize,H1+1);
% %         h1new(:,1) = 1;
% %         h1new(:,2:H1+1) = h1(:,:);
% %        
% %         dropoutExamplesh1 = randperm(H1+1);
% %         for i=1:fix(dropoutPercent*(H1+1))
% %             h1new(:,dropoutExamplesh1(i))=0;
% %         end
% %         
% %         
% %         %h2= batchSize x H2
% %         h2 = h1new*transpose(v1);
% %         
% %         %h2=arrayfun(@sigmoid,h2);
% %         
% %         h2=sigmf(h2,[1 0]);
% %        
% %         h2new = zeros(batchSize,H2+1);
% %         h2new(:,1) = 1;
% %         h2new(:,2:H2+1) = h2(:,:);
% %         
% %         dropoutExamplesh2 = randperm(H2+1);
% %         for i=1:fix(dropoutPercent*(H2+1))
% %             h2new(:,dropoutExamplesh2(i))=0;
% %             
% %         end
% %        
% %         %o= batchSize x K
% %         o=zeros(batchSize,K);
% %         o = h2new*transpose(v2);
% %        
% %          if n==1
% %         output= o(1,1);
% %     end
% %        
% %         
% %         % ---------
% %         
% %         % backward pass
% % 
% %             %minibatch method
% %         % delta3 = batchSize x K
% %         
% %         
% %             delta3=o-trainYnew;   % batchsize x K
% %             
% %             delta2=(delta3*v2).*(h2new).*(1-h2new);  %  batchsize x (H2+1)
% %             delta2=delta2(:,2:size(delta2,2));    %  batchsize x (H2)
% %             
% %             delta1=(delta2*v1).*h1new.*(1-h1new);
% %             delta1=delta1(:,2:size(delta1,2));  % batchsize x H1
% %             
% %             deltav2= eta*delta3'*h2new;
% %             deltav1= eta*delta2'*h1new;
% %             deltaw= eta*delta1'*trainXnew;
% %         
% %         
% % %    delta3=o-trainYnew;
% % %         
% % %    delta2=(delta3*v2).*(h2new.*(1-h2new));
% % %    
% % %  
% % %     % delta2 = batchSize x H2
% % %    delta2=delta2(:,2:size(delta2,2));
% % %    
% % %    delta1=(delta2*v1).*(h1new.*(1-h1new));
% % %             delta1=delta1(:,2:size(delta1,2));  % batchsize x H1
% % %    
% % %    
% % %    
% % %             
% % %             deltav2= transpose(delta3)*h2new;
% % %             deltav1= transpose(delta2)*h1new;
% % %             deltaw= transpose(delta1)*trainXnew;
% % 
% %             w=w-(deltaw/batchSize);
% %             v1=v1-(deltav1/batchSize);
% %             v2=v2-(deltav2/batchSize);
% %           
% %       start=start+batchSize;  
% %     end
% %     
% %     
% %     ydash = mlptest(trainX, w, v1,v2,dropoutPercent);
% %     % compute the training error
% %     % ---------
% %     %'TO DO'% uncomment the next line after adding the necessary code
% %     trainerror(epoch) = 0;
% %     for i=1:n1
% %             trainY(i,1);
% %             ydash(i,1);
% %             trainerror(epoch) = trainerror(epoch) + 0.5*((trainY(i,1)-ydash(i,1)).^2);
% %     end
% %     trainerror(epoch) = trainerror(epoch)/n1;
% %     % ---------
% %     disp(sprintf('training error after epoch %d: %f\n',epoch,...
% %         trainerror(epoch)));
% %     
% %     
% %     ydash2 = mlptest(testX, w, v1,v2,dropoutPercent);
% %     
% %     testerror(epoch) = 0;
% %     for i=1:n2
% %             testY(i,1);
% %             ydash2(i,1);
% %             testerror(epoch) = testerror(epoch) + 0.5*((testY(i,1)-ydash2(i,1)).^2);
% %     end
% %     testerror(epoch) = testerror(epoch)/n2;
% %     % ---------
% %     disp(sprintf('Validation error after epoch %d: %f\n',epoch,...
% %         testerror(epoch)));
% % end
% % 
% % Xlinspace = 1:1:nEpochs;
% % plot(Xlinspace,trainerror,'linewidth',2,'color','k')
% % hold on;
% % plot(Xlinspace,testerror,'linewidth',2,'color','r')
% % legend('Train error', 'Validation error', 'Location', 'NorthOutside', ...
% %     'Orientation', 'horizontal');
% % 
% % return;
% % 
% % function ydash = mlptest(X, w, v1,v2,dropoutPercent)
% % % forward pass of the network
% % N = size(X,1);
% % D = size(X,2);
% % H1 = size(w,1);
% % H2 = size(v1,1);
% % % number of inputs
% % 
% % 
% % % number of outputs
% % K = size(v2,1);
% % 
% % % forward pass to estimate the outputs
% % % --------------------------------------
% % % input to hidden for all the data points
% % % calculate the output of the hidden layer units
% % % ---------
% % %'TO DO'%
% % h1 = (1-dropoutPercent)*X*transpose(w);
% % 
% % 
% %    % h1=arrayfun(@sigmoid,h1);
% %     h1=sigmf(h1,[1 0]);
% % 
% % 
% % h1new = zeros(N,H1+1);
% % h1new(:,1) = 1;
% % h1new(:,2:H1+1) = h1(:,:);
% % 
% % h2 = (1-dropoutPercent)*h1new*transpose(v1);
% % 
% %    % h2=arrayfun(@sigmoid,h2);
% %     
% %     h2=sigmf(h2,[1 0]);
% % 
% % h2new = zeros(N,H2+1);
% % h2new(:,1) = 1;
% % h2new(:,2:H2+1) = h2(:,:);
% % 
% % o = (1-dropoutPercent)*h2new*transpose(v2);
% % 
% % ydash = o;
% % 
% % return;