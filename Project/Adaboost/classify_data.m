function [error_rate, false_pos_ratio] = classify_data(X_test, Y_test, T, ...
          ada_alpha_vec, ada_theta_vec, ada_sign_vec, ada_attr_vec)
    false_neg = 0;
    false_pos = 0;
    for i = 1:size(X_test,1)
        sample = X_test(i,:);
        
        attr_t_mat = zeros(size(X_test, 2), T);
        offsets = ada_attr_vec + (0:T-1)' .* size(X_test, 2);
        attr_t_mat(offsets) = 1;
        T_sample = sample * attr_t_mat;
        T_sample = transpose(T_sample);
        
        T_classif = -ada_sign_vec;
        above_theta = T_sample > ada_theta_vec;
        T_classif(above_theta) = ada_sign_vec(above_theta);
        
        classif_sum = sum(T_classif .* ada_alpha_vec);        
        classif = sign(classif_sum);
        false_neg = false_neg + (classif==-1 && Y_test(i)==+1);
        false_pos = false_pos + (classif==+1 && Y_test(i)==-1);
    end
    error_rate = (false_neg + false_pos) / size(X_test,1);    
    num_of_hams = sum(Y_test == -1);
    false_pos_ratio = false_pos / num_of_hams;
 end
