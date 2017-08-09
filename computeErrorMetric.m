function J = computeErrorMetric(X, y, theta1, theta2)

m = length(y);
J = (sum(sum(((predict(theta1, theta2, X)) - y) .^ 2))) / (2 * m);

end
