clc;
clear;
% Read the file
filename = 'IrisData.txt';  
data = readtable(filename, 'Delimiter', ',', 'ReadVariableNames', false);
numericData = data(:, 1:4);
label_data = data(:, 5);
labels = zeros(height(data), 3);% create a array to store one-hot labels

% Convert the label data from words to ont-hot encoding
setosa_index = find(strcmp(data{:, 5}, 'Iris-setosa'));
versicolor_index = find(strcmp(data{:, 5}, 'Iris-versicolor'));
virginica_index = find(strcmp(data{:, 5}, 'Iris-virginica'));
labels(setosa_index, 1) = 0.8;
labels(setosa_index, 2:3) = 0.2;
labels(versicolor_index, 1) = 0.2;
labels(versicolor_index, 2) = 0.8;
labels(versicolor_index, 3) = 0.2;
labels(virginica_index, 1:2) = 0.2;
labels(virginica_index, 3) = 0.8;



% Scrambled dataset and randomly pick 70% as training data, 30% as
% validation data
numSamples = size(data,1);
randomIndices = randperm(numSamples);
numTrainSamples  = round(0.7 * numSamples);
trainIndices = randomIndices(1:numTrainSamples);
testIndices = randomIndices(numTrainSamples+1:end);

% Get the training data and validation data
trainData = numericData(trainIndices, :);
trainLabels = labels(trainIndices, :);
validationData = numericData(testIndices, :);
validationLabels = labels(testIndices, :);
trainData = table2array(trainData);% Convert the input data from table to array
validationData = table2array(validationData);



% Train the neural network by toolbox
%net = newff(minmax(trainData),[4,5,3,3],{'tansig','tansig', 'tansig', 'tansig'},'Traingd');
net = feedforwardnet([4, 5, 3, 3]);  

% Set activate function and train function
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'tansig';
net.trainFcn = 'traingd';

% Set training parameter
net.trainParam.show = 2000;
net.trainParam.lr = 0.01;
net.trainParam.epochs = 10000;
net.trainParam.goal = 1e-4;


[net, tr] = train(net, trainData', trainLabels');


% Predict using validation data
net_output = sim(net, validationData');


% Convert actual and predicted outputs to class indices
validationLabels = validationLabels';
[p, q] = size(validationLabels);
idx_Actual = zeros(1, q);
idx_Predicted = zeros(1, q);

for j = 1:q
    [~, idx_Actual(j)] = max(validationLabels(:, j));
    [~, idx_Predicted(j)] = max(net_output(:, j));
end

% Plot actual vs predicted values
figure;
plot(idx_Actual, 'b-', 'LineWidth', 1.5);
hold on;
plot(idx_Predicted, 'r-', 'LineWidth', 1.5);
hold off;
legend('Actual', 'Predicted');
title('Actual vs Predicted Classification');

% Plot error
figure;
plot(abs(idx_Actual - idx_Predicted), 'ro');
title('Prediction Error');

% Generate and display the confusion matrix
confMat = confusionmat(idx_Actual, idx_Predicted);
disp('Confusion Matrix:');
disp(confMat);

% Plot the confusion matrix for better visualization
figure;
confusionchart(idx_Actual, idx_Predicted);
title('Confusion Matrix for Validation Data');

% Calculate precision for each class
precision = diag(confMat) ./ sum(confMat, 2);
disp('Precision for each class:');
disp(precision);
