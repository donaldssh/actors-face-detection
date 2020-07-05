clear all;
close all;

%load library for face alignment
addpath('func');

%Npeople is the number of subjects
Npeople = 13;

%Set the path to the train and test set
train_images = '../croppedfaces'
test_images = '../croppedfacesTest'

%load training data
im = imageDatastore(train_images,'IncludeSubfolders',true,'LabelSource','foldernames');
% Resize the images to the input size of the net
im.ReadFcn = @(loc)imresize(imread(loc),[64,64]);
%Split the training set into training (80%) and validation (20%) 
[Train ,Validation] = splitEachLabel(im,0.8,'randomized');
%Load test data
Test = imageDatastore(test_images,'IncludeSubfolders',true,'LabelSource','foldernames');
% Resize the images to the input size of the net
Test.ReadFcn = @(loc)imresize(imread(loc),[64,64]);
 
 
%Network structure updated from from lab3
%Sequence of layers for the network
layer_vet=[
    imageInputLayer([64 64 3])
    
    convolution2dLayer([3 3],64)
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer([5 5],128);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer([8 8],128);
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(9, 128, 'Padding','same');
    batchNormalizationLayer
    reluLayer();
    maxPooling2dLayer(2,'Stride',2)
    dropoutLayer(0.25)

    fullyConnectedLayer(Npeople)

    softmaxLayer();

    classificationLayer()
 ];
 

% options for training the net if your newnet performance is low decrease
% the learning_rate
learning_rate = 0.00005;
opts = trainingOptions("rmsprop","InitialLearnRate",learning_rate,...
    'MaxEpochs',10,...
    'MiniBatchSize',128,...
    'ValidationData',Validation,...
    'ExecutionEnvironment','gpu',...
    'Plots','training-progress');

%training networks
[newnet,info] = trainNetwork(Train, layer_vet, opts);

%predict the labels for the test set
[predict,scores] = classify(newnet,Test);

%measure the accuracy
names = Test.Labels;
pred = (predict==names);
s = size(pred);
acc = sum(pred)/s(1);
fprintf('The accuracy of the test set is %f %% \n',acc*100);

nntraintool close
%plot confusion matrix
plotconfusion(names, predict);

%save the network 
save CNNNet0107_2 newnet

% export to onnx format to import it on python/opencv
filename = 'CNNNet0107_2.onnx';
exportONNXNetwork(newnet,filename)
  
  
