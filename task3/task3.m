
%% hyper-parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
datasetPath = "MixDataset";
% choose from "Alex" and "Google"
backbone = "Google";
lr = 1e-4;
n_epoch = 5;
batch_size = 20;

%% load dataset >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
imds = imageDatastore(datasetPath,"IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%% get classes for training set >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
classNames = categories(imdsTrain.Labels); % get the names of classes
numClasses = numel(classNames); % get the number of classes

%% load pre-trained network >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if backbone == "Google" % load GoogLeNet
    net = imagePretrainedNetwork("googlenet",NumClasses=numClasses);
    disp("GoogLeNet for transfer backbone.")
elseif backbone == "Alex" % load AlexNet
    net = imagePretrainedNetwork("alexnet",NumClasses=numClasses);
    disp("AlexNet for transfer backbone.")
end

%% get the input size of network
inputSize = net.Layers(1).InputSize;

%% open the visual analyzer
analyzeNetwork(net)


%% apply data augmentation to avoid overfitting
randomPixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',randomPixelRange, ...
    'RandYTranslation',randomPixelRange);

%% resize training images
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);

%% auto-resize validation images
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%% training options
options = trainingOptions("rmsprop", ...
    MiniBatchSize=batch_size, ...
    MaxEpochs=n_epoch, ...
    Metrics="accuracy", ...
    InitialLearnRate=lr, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=3, ...
    Verbose=false, ...
    Plots="training-progress");

%% train net with cross entropy loss
net = trainnet(augimdsTrain,net,"crossentropy",options);

%% get four predictions and show them up
scores = minibatchpredict(net,augimdsValidation);
YPred = scores2label(scores,classNames);

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end


