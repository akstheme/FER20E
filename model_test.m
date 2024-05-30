clc;
clear;
close all;
% Set your dataset paths
datasetLocation = 'E:\IITD_ResearchWork\Database\IITD_FER_Augmented';
load('Trained_Mobilenet22.mat')
% Create an imageDatastore to load images
imds = imageDatastore(datasetLocation, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Resize images to 128x128
% targetSize = [294, 294, 3];
targetSize = [227, 227, 3];


% Split the dataset into training, validation, and testing sets (80% - 10% - 10%)
[imds_Train, imds_Test] = splitEachLabel(imds, 0.7, 0.3, 'randomized');
imdsTrain = augmentedImageDatastore(targetSize, imds_Train);
% imdsValidation= augmentedImageDatastore(targetSize, imds_Validation);
imdsTest= augmentedImageDatastore(targetSize, imds_Test);


% Define the custom deep learning architecture
% inputSize = [224 224 3];


% Train the network
net = trainedNetwork_1;

% Evaluate the trained network on the test set
YPred = classify(net, imdsTest);
YTest = imds_Test.Labels;

% Calculate the accuracy
accuracy = mean(YPred == YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

plotconfusion(YTest,YPred)
