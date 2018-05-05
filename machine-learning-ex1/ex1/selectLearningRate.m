function[theta, J_history] = selectLearningRate(X,y)
% Choose some alpha value
alpha = 0.01;
num_iters = 400;

alpha_1 = 0.001;
num_iters_1 = 400;

alpha_2 = 0.1;
num_iters_2 = 30;

% 测试得到的下降最快的learning rate就是１，但是theta_histroy已经开始来回晃了
alpha_3 = 1;
num_iters_3 = 10;

alpha_4 = 1.2;
num_iters_4 = 10;

alpha_5 = 1.5;
num_iters_5 = 10;



% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history, theta2_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

theta_1 = zeros(3, 1);
[theta_1, J_history_1, theta2_history_1] = gradientDescentMulti(X, y, theta_1, alpha_1, num_iters_1);

theta_2 = zeros(3, 1);
[theta_2, J_history_2, theta2_history_2] = gradientDescentMulti(X, y, theta_2, alpha_2, num_iters_2);

theta_3 = zeros(3, 1);
[theta_3, J_history_3, theta2_history_3] = gradientDescentMulti(X, y, theta_3, alpha_3, num_iters_3);

theta_4 = zeros(3, 1);
[theta_4, J_history_4, theta2_history_4] = gradientDescentMulti(X, y, theta_4, alpha_4, num_iters_4);

theta_5 = zeros(3, 1);
[theta_5, J_history_5, theta2_history_5] = gradientDescentMulti(X, y, theta_5, alpha_5, num_iters_5);


% Plot the J(theta(2))
figure;
hold on;
%plot(theta2_history, J_history, '-b', 'LineWidth', 2);
%plot(theta2_history_1, J_history_1, '-r', 'LineWidth', 2);
plot(theta2_history_2, J_history_2, '-k', 'LineWidth', 2);
plot(theta2_history_3, J_history_3, '-y', 'LineWidth', 2);
plot(theta2_history_4, J_history_4, '-g', 'LineWidth', 2);
legend('0.1','1','1.2');
% legend('0.01','0.001','0.1','1','1.2');
xlabel('theta(2)')
ylabel('J(theta)')
title('J(theta)')

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
hold on;
plot(1:numel(J_history_1), J_history_1, '-r', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
legend('0.01','0.001');


figure;
plot(1:numel(J_history_2), J_history_2, '-k', 'LineWidth', 2);
hold on;
plot(1:numel(J_history_3), J_history_3, '-r', 'LineWidth', 2);
plot(1:numel(J_history_4), J_history_4, '-y', 'LineWidth', 2);
axis([0 10])
xlabel('Number of iterations');
ylabel('Cost J');
legend('0.1','1','1.2');


figure;
plot(1:numel(J_history_5), J_history_5, '-b', 'LineWidth', 2);
hold on;
xlabel('Number of iterations');
ylabel('Cost J');
legend('1.5');



% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');