function p = predict(theta1, theta2, X)

m = size(X, 1);
num_labels = size(theta2, 1);
 p = zeros(m, 1);

h1 = sigmoid([ones(m, 1) X] * theta1');
p = sigmoid([ones(m, 1) h1] * theta2');

end