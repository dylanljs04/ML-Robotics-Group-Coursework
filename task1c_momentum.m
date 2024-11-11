
clear

lr = 0.001;
rho = 0.8;
n_iter = 5000;
n_hidden = 3;


input = -1:0.05:1;

expected_output = 0.8*input.^3 + 0.3*input.^2 - 0.4 * input;
hidden_w = rand([1,n_hidden])*2-1;
hidden_b = rand([1,n_hidden])*2-1;

output_w = rand(1,n_hidden)*2-1;
output_b = rand(1)*2-1;

% initialize the size of hidden output 
hidden_output = zeros([1,n_hidden]);

% initialize the size of velocities
output_w_velocity = zeros([1,length(output_w)]);
output_b_velocity = 0;
hidden_w_velocity = zeros([1,length(hidden_w)]);
hidden_b_velocity = 0;

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
        
        % apply SGD + momentum

        output_w_velocity = rho*output_w_velocity + output_w_grad;
        output_b_velocity = rho*output_b_velocity + output_b_grad;
        hidden_w_velocity = rho*hidden_w_velocity + hidden_w_grad;
        hidden_b_velocity = rho*hidden_b_velocity + hidden_b_grad;
        
        % step
        output_w = output_w - lr * output_w_velocity;
        output_b = output_b - lr * output_b_velocity;
        hidden_w = hidden_w - lr * hidden_w_velocity;
        hidden_b = hidden_b - lr * hidden_b_velocity;
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
figure(2)
subplot(2,1,1)
plot(loss_list)
subplot(2,1,2)
plot(input,model_test_output, "-ob")
hold on
plot(input,expected_output, "-r")
hold off
