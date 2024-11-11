
clear

lr = 0.001;
decay_constant = 0.99;
n_iter = 5000;
n_hidden = 3;
epsilon = 1e-9;

input = -1:0.05:1;
expected_output = 0.8*input.^3 + 0.3*input.^2 - 0.4 * input;
hidden_w = rand([1,n_hidden])*2-1;
hidden_b = rand([1,n_hidden])*2-1;

output_w = rand(1,n_hidden)*2-1;
output_b = rand(1)*2-1;

% initialize the size of hidden output 
hidden_output = zeros([1,n_hidden]);

% initialize the size of v
output_w_v = zeros([1,length(output_w)]);
output_b_v = 0;
hidden_w_v = zeros([1,length(hidden_w)]);
hidden_b_v = 0;

% training loop
for i = 1:n_iter

    % training by each input in an iteration
    for j = 1:length(input)

        % forward proporgation for hidden layer
        hidden_output = tanh(input(j) * hidden_w + hidden_b);
        
        % forward proporgation for output layer
        model_output = sum(hidden_output.*output_w) + output_b;

        % loss
        loss = 1/2*(input(j)-model_output)^2;

        % back proporgation

        % evaluate grads
        output_w_grad = (model_output-expected_output(j))*tanh(hidden_w*input(j)+hidden_b);
        output_b_grad = (model_output-expected_output(j));
        hidden_w_grad = (model_output-expected_output(j))*output_w.*(1-tanh(hidden_w*input(j)+hidden_b).^2)*input(j);
        hidden_b_grad = (model_output-expected_output(j))*output_w.*(1-tanh(hidden_w*input(j)+hidden_b).^2);
        
        % apply RMS Prop

        output_w_v = decay_constant*output_w_v + (1-decay_constant)*output_w_grad.^2;
        output_b_v = decay_constant*output_b_v + (1-decay_constant)*output_b_grad.^2;
        hidden_w_v = decay_constant*hidden_w_v + (1-decay_constant)*hidden_w_grad.^2;
        hidden_b_v = decay_constant*hidden_b_v + (1-decay_constant)*hidden_b_grad.^2;

        % step
        output_w = output_w - lr./sqrt(output_w_v+epsilon) .* output_w_grad;
        output_b = output_b - lr./sqrt(output_b_v+epsilon) .* output_b_grad;
        hidden_w = hidden_w - lr./sqrt(hidden_w_v+epsilon) .* hidden_w_grad;
        hidden_b = hidden_b - lr./sqrt(hidden_b_v+epsilon) .* hidden_b_grad;

    end
    if mod(i,50) == 0
        loss_list(i/50) = loss;
    end
end


% test the model

model_test_output = zeros([1,length(input)]);

for j = 1:length(input)
    % forward proporgation for hidden layer
    hidden_output = tanh(input(j) * hidden_w + hidden_b);
    
    % forward proporgation for output layer
    model_output = sum(hidden_output.*output_w) + output_b;

    model_test_output(j) = model_output;
end

% plot(input,model_test_output, "-ob")
figure(3)
subplot(2,1,1)
plot(loss_list)
subplot(2,1,2)
plot(input,model_test_output, "-ob")
hold on
plot(input,expected_output, "-r")
hold off

