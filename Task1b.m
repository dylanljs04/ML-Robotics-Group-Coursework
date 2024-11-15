clear all
close all 
clc 

%% Generate training data
x = -1:0.05:1; % Input data as a row vector
len = length(x);

% Training data output with added noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);
figure, plot(x, d,'k+'), title('Training Data for Regression')

% Test data
xtest = -0.97:0.1:0.93;

%% Settings
lr = 0.1; % Learning rate
n_iter = 2000; % Number of iterations
n_hidden = 3; % Number of hidden layer nodes

% Initialize weights and biases with random numbers
w_input_hidden = randn(1, n_hidden); % Weights from input to hidden layer (1x3 double)
w_hidden_output = randn(n_hidden, 1); % Weights from hidden layer to output (3x1 double)
b_hidden = randn(1, n_hidden); % Biases for hidden layer (1x3 double)
b_output = randn; % Bias for output layer

%% Training process using stochastic gradient descent (SGD)
losses = zeros(1, n_iter); % Track loss during training
for i = 1:n_iter % Loop for number of iterations
    total_loss = 0; % initialise loss for this iteration
    for j = 1:len % Loop through length of input data
        % Forward Pass
        % Current input and target
        curr_input = x(j);
        target = d(j); 
        % Hidden Layer
        hidden_input = curr_input * w_input_hidden + b_hidden; % Weighted sum + Bias
        hidden_output = tanh(hidden_input); % Activation Function
        % Output Layer
        output = hidden_output * w_hidden_output + b_output; % Weighted Sum + Bias
        % Error 
        error = target - output; 
        total_loss = total_loss + 0.5 * error^2; % Loss
        % Backward Pass
        % Output Layer and Hidden Layer Gradients
        grad_output = error * -1; 
        grad_hidden = (grad_output * w_hidden_output').*(1-hidden_output.^2);
        % Update Weights and Biases
        w_hidden_output = w_hidden_output - lr*(grad_output * hidden_output');
        w_input_hidden = w_input_hidden - lr*(grad_hidden * curr_input);
        b_output = b_output - lr*grad_output;
        b_hidden = b_hidden - lr*grad_hidden;
    end
    % Store Average loss for iteration
    losses(i) = total_loss/len;
end

%% Test the neural network
net_output = zeros(size(xtest));
for i = 1:length(xtest)
    hidden_input = xtest(i) * w_input_hidden + b_hidden;
    hidden_output = tanh(hidden_input);
    net_output(i) = hidden_output * w_hidden_output + b_output;
end

%% Plot out the test results
hold on;
plot(xtest, net_output, 'r-');
legend('Training Data', 'Test Prediction')

%% Plot the loss over iterations
figure;
plot(1:n_iter, losses, 'b-');
title('Loss over Iterations');
xlabel('Iteration');
ylabel('Loss');