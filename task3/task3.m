imds = imageDatastore("DataSet","IncludeSubfolders",true,"LabelSource","foldernames");
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

% get labels for training set
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

% load pre-trained network
net = imagePretrainedNetwork("googlenet",NumClasses=numClasses);
% net = imagePretrainedNetwork("alexnet",NumClasses=numClasses);
net = setLearnRateFactor(net,"fc8/Weights",20);
net = setLearnRateFactor(net,"fc8/Bias",20);

% get the input size of network
inputSize = net.Layers(1).InputSize;

% open the visual analyzer
analyzeNetwork(net)
 

% apply data augmentation to avoid overfitting
randomPixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',randomPixelRange, ...
    'RandYTranslation',randomPixelRange);

% resize training images
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

% auto-resize validation images
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions("sgdm", ...
    MiniBatchSize=10, ...
    MaxEpochs=6, ...
    Metrics="accuracy", ...
    InitialLearnRate=1e-4, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=3, ...
    Verbose=false, ...
    Plots="training-progress");

net = trainnet(augimdsTrain,net,"crossentropy",options);

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


