function [trainError, testError, testFalsePos] = ...
    adaboost(T,trainData, testData, trainFrac, plotConverge)


%target
Y = 2*trainData(:,end)-1;
testY = 2*testData(:,end)-1;
%input features
X = trainData(:,1:end-1);
testX = testData(:,1:end-1);

%init with uniform distribution
distrib = (1/size(X, 1)) .* ones(size(X, 1),1);

false_pos_vec = zeros(T, 1);
ada_alpha_vec = zeros(T, 1);
ada_theta_vec = zeros(T, 1);
ada_sign_vec = zeros(T, 1);
ada_attr_vec = zeros(T, 1);
ada_test_error = zeros(T,1);
ada_train_error = zeros(T,1);
attrs = 1:size(X, 2);

for t = 1:T
    display(t);
    [error_rate_vec, theta_vec, sign_vec] = arrayfun(@(attr)...
        weak_learn(X, distrib, Y, attr), attrs);
    [min_error, min_error_idx] = min(error_rate_vec);
    min_error_theta = theta_vec(min_error_idx);
    min_error_sign = sign_vec(min_error_idx);
    min_error_attr = attrs(min_error_idx);
    alpha_t = 0.5*log((1-min_error)/min_error);

    %classify the training set using the best hypothesis
    h_classif = get_hypothesis_classification( ...
        X, min_error_theta, min_error_sign, min_error_attr);

    %update the training set distribution
    distrib = exp(-alpha_t .* h_classif .* Y) .* distrib;	
    %normalize
    distrib = distrib ./ sum(distrib);
	
    ada_alpha_vec(t) = alpha_t;
    ada_theta_vec(t) = min_error_theta;
    ada_sign_vec(t) = min_error_sign;
    ada_attr_vec(t) = min_error_attr;
    
    [ada_test_error(t),false_pos_vec(t)] = classify_data(...
        testX, testY, t, ada_alpha_vec(1:t), ada_theta_vec(1:t),...
        ada_sign_vec(1:t), ada_attr_vec(1:t));
    [ada_train_error(t),dummy] = classify_data(...
        X, Y, t, ada_alpha_vec(1:t), ada_theta_vec(1:t),...
        ada_sign_vec(1:t), ada_attr_vec(1:t));
end
if plotConverge
    h=figure;
    hold on;
    plot(ada_test_error, 'r-o');
    plot(ada_train_error, 'b-o');
    plot(false_pos_vec, 'g-o');
    txt = sprintf('%g%% of train-set ,  T = %d', trainFrac*100, T);
    title(txt);
    legend('Testing', 'Training', 'Test false pos');
    txt = sprintf('error_for_T_%d_train_percent_%g.fig', T, trainFrac*100);
    saveas(h, txt);
end
trainError = ada_train_error(end)
testError = ada_test_error(end)
testFalsePos = false_pos_vec(end)
end

function h_classif = get_hypothesis_classification(x_train, theta, sign, attr)
    h_classif = -sign .* ones(size(x_train,1),1);
    h_classif(x_train(:, attr) > theta) = sign;
end