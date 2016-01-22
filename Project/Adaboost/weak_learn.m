%weak_learn predicts the classification of the training set using a
%parameter theta, in the following manner:
%Classify as +1 iff x_train(index_i, index_j) > theta
function [error_rate, theta, sign]  = weak_learn( ...
    x_train, distrib, y_train, i_feature)

    %assert(sum(disrib) == 1,'distribution of x_train does not sum to 1');
    train_data = x_train(:, i_feature);
    %sort the data acording to the feature value
    [sorted_data,i_vec] = sort(train_data);
    distrib = distrib(i_vec);
    y_train = y_train(i_vec);

    thresh_vec = unique(sorted_data);
    [error_rate_vec, sign_vec] = arrayfun(@(thresh) ...
        calc_error(sorted_data, distrib, y_train, thresh), thresh_vec);
    [error_rate,i] = min(error_rate_vec);
    theta = thresh_vec(i);
    sign = sign_vec(i);
end

function [error_rate, sign] = calc_error( ...
    sorted_data, distrib, classif, thresh)

    below_th = sorted_data <= thresh;
    above_th = ~below_th;
    
    minus_classif = classif == -1;
    plus_classif = ~minus_classif;
     
    %classify using the rule h(x) = -1 iff x > thresh
    misclassif_neg_above = ...
        (minus_classif & below_th)  | (plus_classif & above_th);
    %classify using the rule h(x) = -1 iff x <= thresh
    misclassif_neg_below = ...
        (plus_classif & below_th)  | (minus_classif & above_th);
    
    neg_above_err = sum(distrib(misclassif_neg_above));
    neg_below_err = sum(distrib(misclassif_neg_below));
    
    if neg_above_err < neg_below_err
        sign = -1;
        error_rate = neg_above_err;
    else
        sign = 1;
        error_rate = neg_below_err;
    end    
end