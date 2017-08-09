
clear ; %clears all the variables 
close all; %closes extra windows
clc %clears the screen

% ====================================================== Part 2: NEURAL NETWORKS  ======================================================

% ---------------------------------------------------(a)------------------------------------------------------
trainD = csvread('train.data');
yTrain = trainD(:,size(trainD,2));
xTrain = trainD(:,1:size(trainD,2)-1);
mTrain = length(yTrain);         %no. of training examples
X = [ones(mTrain, 1), xTrain]; % Add a column of ones to x (intercept term)
n = size(xTrain,2);

input_layer_size  = 126;  % 42 * 3 (triplet values for each of the 42 board positions)
hidden_layer_size = 100;   % 100 hidden units
num_labels = 3;          % 3 labels, from 1 to 3  

yTrain_mat = zeros(mTrain, 3);
for i = 1 : mTrain
    yTrain_mat(i,yTrain(i)) = 1;
end

[theta1 theta2] = varying_units(input_layer_size, hidden_layer_size, num_labels, xTrain, yTrain_mat);
pTrain = predict(theta1, theta2, xTrain);
[dummy, p] = max(pTrain, [], 2);
accuracy_train = (length(find(yTrain == p)) * 100) / mTrain

testD = csvread('test.data');
yTest = testD(:,size(testD,2));
xTest = testD(:,1:size(testD,2)-1);
mTest = length(yTest);
pTest = predict(theta1, theta2, xTest);
[dummy, p] = max(pTest, [], 2);
accuracy_test = (length(find(yTest == p)) * 100) / mTest
pause;
% ---------------------------------------------------(b)------------------------------------------------------

theta1 = (-1 + (1+1) * rand(hidden_layer_size, input_layer_size + 1)) / 1000;		%generating random small nos. between -1 and 1
theta2 = (-1 + (1+1) * rand(num_labels, hidden_layer_size + 1)) / 1000;

J = [0; 0];
condition = true; %for implementing a do-while loop
epsilon = 0.001;
iter = 0;
tic
while condition
    iter = iter + 1;
	eta = 0.2 / sqrt(iter)
	J(1) = J(2);
    J(2) = computeErrorMetric(xTrain, yTrain_mat, theta1, theta2)
	for i = 1 : mTrain
		x = X(i,:)';
		hidden_output = sigmoid(theta1 * x);
		hidden_output = [1; hidden_output];
		final_output = sigmoid(theta2 * hidden_output);
		t = yTrain_mat(i,:)';
		delta_o = final_output .* (1 - final_output) .* (t - final_output);
		delta_h = hidden_output .* (1 - hidden_output) .* (theta2' * delta_o);
		D_theta1 = eta * delta_h(2:length(delta_h),:) * x';
		D_theta2 = eta * delta_o * hidden_output';
		theta1 = theta1 + D_theta1;
		theta2 = theta2 + D_theta2;
    end
	condition = (abs(J(1) - J(2)) > epsilon); %convergence criteria
end
toc;

pTrain = predict(theta1, theta2, xTrain);
[dummy, p] = max(pTrain, [], 2);
accuracy_train = (length(find(yTrain == p)) * 100) / mTrain

pTest = predict(theta1, theta2, xTest);
[dummy, p] = max(pTest, [], 2);
accuracy_test = (length(find(yTest == p)) * 100) / mTest

pause;

% ---------------------------------------------------(c)------------------------------------------------------
tic;
units = [50:50:500];
accuracy_test = zeros(1,length(units));
for i = 1 : length(units)
	[theta1 theta2] = varying_units(input_layer_size, units(i), num_labels, xTrain, yTrain_mat);
	pTest = predict(theta1, theta2, xTest);
	[dummy, p] = max(pTest, [], 2);
	accuracy_test(i) = (length(find(yTest == p)) * 100) / mTest;
end
accuracy_test
figure;
plot(units, accuracy_test, 'r-');
axis([50 500 0 100])
xlabel('Hidden Layer Units');
ylabel('Accuracy over the Test Set');
toc;
pause;
% ---------------------------------------------------(d)------------------------------------------------------

theta1 = (-1 + (1+1) * rand(hidden_layer_size, input_layer_size + 1)) / 1000;		%generating random small nos. between -1 and 1
theta2 = (-1 + (1+1) * rand(num_labels, hidden_layer_size + 1)) / 1000;
eta = 0.1;
epsilon = 0.001;
J = [0; 0];
condition = true; %for implementing a do-while loop
tic;
iter = 0;
while condition
	J(1) = J(2);
	J(2) = computeErrorMetric_soft(xTrain, yTrain_mat, theta1, theta2)
	for i = 1 : mTrain
		x = X(i,:)';
		hidden_output = softplus(theta1 * x);
		hidden_output = [1; hidden_output];
		final_output = softplus(theta2 * hidden_output);
		t = yTrain_mat(i,:)';
		delta_o = (1 - exp(-final_output)) .* (t - final_output);
		delta_h = (1 - exp(-hidden_output)) .* (theta2' * delta_o);
		D_theta1 = eta * delta_h(2:length(delta_h),:) * x';
		D_theta2 = eta * delta_o * hidden_output';
		theta1 = theta1 + D_theta1;
		theta2 = theta2 + D_theta2;
	end
	condition = (abs(J(1) - J(2)) > epsilon); %convergence criteria
end
toc;

pTrain = predict_soft(theta1, theta2, xTrain);
[dummy, p] = max(pTrain, [], 2);
accuracy_train = (length(find(yTrain == p)) * 100) / mTrain

pTest = predict_soft(theta1, theta2, xTest);
[dummy, p] = max(pTest, [], 2);
accuracy_test = (length(find(yTest == p)) * 100) / mTest
