function [theta1 theta2] = varying_units(input_layer_size, hidden_layer_size, num_labels, xTrain, yTrain_mat)
	
	theta1 = (-1 + (1+1) * rand(hidden_layer_size, input_layer_size + 1)) / 1000;		%generating random small nos. between -1 and 1
	theta2 = (-1 + (1+1) * rand(num_labels, hidden_layer_size + 1)) / 1000;
	
	mTrain = length(yTrain_mat);         %no. of training examples
	X = [ones(mTrain, 1), xTrain]; % Add a column of ones to x (intercept term)
	eta = 0.07;
	epsilon = 0.001;
	J = [0; 0];
	condition = true; %for implementing a do-while loop
	tic;
	while condition
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