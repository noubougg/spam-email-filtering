% this function expects the labels to be 1 and -1.
% w is the weight vector
function [error_ratio, false_positives_ratio] = hyperplaneClassify(w, thresh, featureVectors, labels)

[num_of_items, num_of_features] = size(featureVectors);

mistakes = 0;
false_positives = 0;

for i = 1:num_of_items
    x = featureVectors(i, :);
    label = labels(i);
        
    decision = sign( dot(w, x) - thresh );
    if decision == 0
        decision = 1;
    end
    
    if decision ~= label
        mistakes = mistakes + 1;
        if label == -1
            false_positives = false_positives + 1;
        end
    end
end

num_of_hams = sum(labels == -1);
error_ratio = mistakes / num_of_items;
false_positives_ratio = false_positives / num_of_hams;
