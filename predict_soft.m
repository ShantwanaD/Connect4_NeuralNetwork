function p = predict_soft(theta1, theta2, X)

m = size(X, 1);
num_labels = size(theta2, 1);
p = zeros(m, 1);

h1 = softplus([ones(m, 1) X] * theta1');
p = softplus([ones(m, 1) h1] * theta2');

end