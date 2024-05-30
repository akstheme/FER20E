% clc;
% clear;
% close all;
% Set your dataset paths
datasetLocation = 'E:\IITD_ResearchWork\Database\data_pre_process\FER_Shuffle';
%load('Trained_Mobilenet22.mat')
% Create an imageDatastore to load images
imds = imageDatastore(datasetLocation, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Resize images to 128x128
targetSize = [224, 224];


% Split the dataset into training, validation, and testing sets (80% - 10% - 10%)
[imds_Train, imds_Validation, imds_Test] = splitEachLabel(imds, 0.4, 0.2, 0.4, 'randomized');
imdsTrain = augmentedImageDatastore(targetSize, imds_Train);
imdsValidation= augmentedImageDatastore(targetSize, imds_Validation);
imdsTest= augmentedImageDatastore(targetSize, imds_Test);


% Define the custom deep learning architecture
% inputSize = [224 224 3];
% numClasses = numel(categories(imdsTrain.Labels));

% lgraph=lgraph_1;
lgraph=net_1;

% Set options for training
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MiniBatchSize',32, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu',...
    Metrics = ["accuracy","fscore"]);

% Train the network
% net = trainNetwork(imdsTrain, lgraph, options);
net = trainnet(imdsTrain, lgraph,"crossentropy", options);
% Evaluate the trained network on the test set
% YPred = classify(net, imdsTest);
YPred = minibatchpredict(net, imdsTest);
YTest = imds_Test.Labels;
classNames = categories(YTest);
YP = onehotdecode(YPred,classNames,2);
% Calculate the accuracy
accuracy = mean(YP == YTest);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

plotconfusion(YTest,YP)
confusionchart(YTest,YP)