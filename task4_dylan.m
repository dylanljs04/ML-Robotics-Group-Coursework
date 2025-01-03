%% Defining Parameters
clc;
clear;

% Parameters
gamma = 0.9;            % Discount factor
alpha = 0.1;            % Learning rate
num_episodes = 100000;    % Number of episodes
max_epsilon = 0.9;      % exploration rate (max)
min_epsilon = 0.1;      % exploration rate (min)
lambda = 0.006;         % Decay Factor
start_state = 1;        % State of the start point (1...12)

%% Calculates the Q-table using the parameters defined 

% Calculates the Q-table using established parameters
q_table = calculate_q_table(gamma, alpha, max_epsilon, min_epsilon, lambda, num_episodes, start_state);
display_q_table(q_table)
display_best_action(q_table)

%% Plotting the epsilon decay 

epsilon_val = zeros(length(num_episodes));
for episode = 1:num_episodes
    epsilon_val(episode) = calc_epsilon(max_epsilon, min_epsilon, lambda, episode);
end

plot(1:num_episodes, epsilon_val);
title("Epsilon Value Over Number of Iterations");
ylabel("Epsilon Value (\epsilon)");
xlabel("Number of Episodes");
set(gca, "FontSize", 14);
grid on;

%% Functions

% Runs through the Q-learning implementation and outputs the Q-table
function q_table = calculate_q_table(gamma, alpha, max_epsilon, min_epsilon, lambda, num_episodes, start_state)
    % Initialise Q-table
    q_table = zeros(12, 4);
    % actions_name = {'Up', 'Right', 'Down', 'Left'};

    
    for episode = 1:num_episodes
        epsilon = calc_epsilon(max_epsilon, min_epsilon, lambda, episode);

        % Resets the start state very episode
        state = start_state;

        % Run until we reach a terminal state
        terminal = false;
        i = 1;
        while ~terminal
            % Select action using epsilon-greedy
            % Actions: 1=Up, 2=Right, 3=Down, 4=Left
            if rand < epsilon
                action = randi(4);  % random action
            else
                [~, action] = max(q_table(state, :));  % greedy action
            end

            % Get next state
            next_state = get_next_state(state, action);
            
            % Get the reward of being in the next state
            if isequal(next_state, 8)
                reward = -10;
                terminal = true;
            elseif isequal(next_state, 12)
                reward = 10;
                terminal = true;
            else
                reward = -1;
            end

            % Calculates the new Q-value and update the table
            sample_q = reward + gamma * max(q_table(next_state, :));
            updated_q = q_table(state,action) + alpha*(sample_q - q_table(state,action));
            q_table(state,action) = updated_q;


            % if i < 4
            %     disp(["Q-table Iteration" episode i])
            %     disp(["updated q" updated_q])
            %     disp([state "to" next_state])
            %     disp(["action" actions_name{action}])
            %     display_q_table(q_table)
            %     i = i+1;
            % end

            % Move to next state
            state = next_state;
        end
        % disp(["Full Q-table" episode])
        % display_q_table(q_table);

    end
end


% ------------------------------------------------------------
% /////////////   HELPER FUNCTIONS  //////////////////
% ------------------------------------------------------------

function epsilon = calc_epsilon(epsilon_max, epsilon_min, lambda, episode)
    epsilon = epsilon_max * exp(-lambda * episode);
    epsilon = max(epsilon, epsilon_min);
end

% Gets the next cell state given the current state and the action the agent
% performs
function next_state = get_next_state(state, action)
    % Convert the current state to [row,col]
    [row, col] = state_to_row_col(state);

    switch action
        case 1 % Up
            row = row - 1;
        case 2 % Right
            col = col + 1;
        case 3 % Down
            row = row + 1;
        case 4 % Left
            col = col - 1;
    end

    if row < 1 || row > 3 || col < 1 || col > 4 || (isequal(row,2) && isequal(col, 2))
        % stay in the same place if moved to state 6 or moved out of bounds
        next_state = state;
    else
        next_state = row_col_to_state(row, col);
    end
end


function [row, col] = state_to_row_col(state)
    % Map each cells state (1..12) to (row,col)
    %   9  10  11  12
    %   5   6   7   8
    %   1   2   3   4
    switch state
        case 1,  row = 3; col = 1;
        case 2,  row = 3; col = 2;
        case 3,  row = 3; col = 3;
        case 4,  row = 3; col = 4;
        case 5,  row = 2; col = 1;
        case 6,  row = 2; col = 2;
        case 7,  row = 2; col = 3;
        case 8,  row = 2; col = 4;
        case 9,  row = 1; col = 1;
        case 10, row = 1; col = 2;
        case 11, row = 1; col = 3;
        case 12, row = 1; col = 4;
    end
end


function state = row_col_to_state(row, col)
    % Inverse of state_to_row_col
    %   9  10  11  12
    %   5   6   7   8
    %   1   2   3   4
    if row==3 && col==1, state=1;
    elseif row==3 && col==2, state=2;
    elseif row==3 && col==3, state=3;
    elseif row==3 && col==4, state=4;
    elseif row==2 && col==1, state=5;
    elseif row==2 && col==2, state=6;
    elseif row==2 && col==3, state=7;
    elseif row==2 && col==4, state=8;
    elseif row==1 && col==1, state=9;
    elseif row==1 && col==2, state=10;
    elseif row==1 && col==3, state=11;
    elseif row==1 && col==4, state=12;
    end
end


% ------------------------------------------------------------
% /////////////   PLOTTING FUNCTIONS  //////////////////
% ------------------------------------------------------------

% Takes in the q_table array as input and display it as a nice table
function display_q_table(q_table)
    
    % Define the words for the left column (states)
    cells = {'Cell 1', 'Cell 2', 'Cell 3', 'Cell 4', ...
                   'Cell 5', 'Cell 6', 'Cell 7', 'Cell 8', ...
                   'Cell 9', 'Cell 10', 'Cell 11', 'Cell 12'};
    
    % Define the top row (action labels)
    actions = {'Up', 'Right', 'Down', 'Left'};
    
    % Create a table with the Q-values
    q_table_formatted = array2table(q_table, ...
        'VariableNames', actions, ...
        'RowNames', cells);
    
    % Display the nicely formatted table
    disp('Q-table:');
    disp(q_table_formatted);
end


% Takes in the q_table array as an input and display the besy action to be
% taken in each column
function display_best_action(q_table)
    % Define action labels
    actions = {'Up', 'Right', 'Down', 'Left'};
    
    % Initialize the 3x4 grid to store the best actions
    best_action = strings(12, 1);
    
    % Determine the best action for each state
    for cell = 1:12
        if all(q_table(cell, :) == 0)
            % Obstacles or terminal states, leave empty
            best_action(cell) = '';
        else
            % Find the index of the maximum Q-value and map to the action labels
            [~, best_action_index] = max(q_table(cell, :));
            best_action(cell) = actions{best_action_index};
        end
    end
    
    % Reshape the best_action array into a 3x4 grid
    best_action_grid = reshape(best_action, [4, 3])'; % Transpose to match grid layout
    
    % Flip the y-axis to align with grid world's layout
    best_action_grid = flipud(best_action_grid);

    % Define the row and col
    row = {'Row 1', 'Row 2', 'Row 3'};
    col = {'Col 1', 'Col 2', 'Col 3', 'Col 4'};
    
    % Create a table with the Q-values
    best_action_grid = array2table(best_action_grid, ...
        'VariableNames', col, ...
        'RowNames', row);
    
    % Display the resulting table
    disp('Best action in each state for the Grid World');
    disp(best_action_grid);
end
