clc;
clear;
close all;

% Load the trained model
load('Trained_Mobilenet22.mat', 'trainedNetwork_1');

% Create a GUI for selecting the user image
[fileName, pathName] = uigetfile({'*.jpg;*.png;*.jpeg','Image Files (*.jpg, *.png, *.jpeg)'; '*.*', 'All Files (*.*)'}, 'Select an image');
if isequal(fileName,0)
    disp('User canceled the operation');
    return;
end
userImagePath = fullfile(pathName, fileName);

% Read the user image
userImage = imread(userImagePath);

% Preprocess the user image to make it compatible with the input size expected by the model
resizedUserImage = imresize(userImage, [294, 294]); % Resize the image to match the target size
inputImage = augmentedImageDatastore([294, 294, 3], resizedUserImage, 'ColorPreprocessing', 'gray2rgb');

% Predict the emotion in the user image
predictedEmotion = classify(trainedNetwork_1, inputImage);

% Display the predicted emotion
fprintf('Predicted Emotion: %s\n', char(predictedEmotion));

% Display the user image
figure;
imshow(userImage);
title('User Image');
