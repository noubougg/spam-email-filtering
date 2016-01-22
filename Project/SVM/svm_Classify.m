function [error_ratio, false_positives_ratio] = svm_Classify(SVMStruct, featureVectors, labels)

[num_of_items, num_of_features] = size(featureVectors);

classifications = svmclassify(SVMStruct, featureVectors);

isError =  classifications ~= labels;
IsFalsePositive =  (classifications == 1) & (labels == -1);
mistakes = sum(isError);
false_positives = sum(IsFalsePositive);

num_of_hams = sum(labels == -1);
error_ratio = mistakes / num_of_items;
false_positives_ratio = false_positives / num_of_hams;
