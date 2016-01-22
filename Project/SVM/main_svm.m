% SVM main script


% different fractions of training data out of the total data
%TRAIN_FRACS = .1:.1:.3;
TRAIN_FRACS = .4;
% number of runs per fraction (each run is with differnt training data
% which is chosen randomly)
RUNS_PER_FRAC = 10;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
DIRNAME ='../Data/enron1';

% error ratio on test set
testErrorMat     = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
% error ratio on training set
%trainErrorMat    = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
% false positive ratio on test set
testFalsePosMat  = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
for iTrainFrac = 1:NUM_TRAIN_FRACS
    trainFrac = TRAIN_FRACS(iTrainFrac)
    for run=1:RUNS_PER_FRAC
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
        
        % randomly mix the vectors in training set
        % -> no need anymore as the script process.py mixes them
        %perm = randperm(size(train, 1));
        %train = train(perm,:);
        
        % the vectors without the labels
        trainVectors = train(:,1:end-1);
        % the lables
        trainLabels = train(:,end);
        % use lables 1,-1 instead of 1,0
        trainLabels = 2*trainLabels - 1;
        
         % the vectors without the labels
        testVectors = test(:,1:end-1);
        % the lables
        testLabels = test(:,end);
        % use lables 1,-1 instead of 1,0
        testLabels = 2*testLabels - 1;
        

        % perform the algorithm
        SVMStruct = svmAlg(trainVectors, trainLabels);
        
        % check on test set
        [testErrorMat(run,iTrainFrac), ...
         testFalsePosMat(run,iTrainFrac)] ...
            = svm_Classify(SVMStruct, testVectors, testLabels);
                
    end
end

%meanTrainErrorMat = mean(trainErrorMat, 1);
meanTestErrorMat = mean(testErrorMat, 1);
meanTestFalsePosMat= mean(testFalsePosMat, 1);

h = figure; 
hold on;
%plot(TRAIN_FRACS,meanTrainErrorMat, 'b-o');
plot(TRAIN_FRACS,meanTestErrorMat, 'r-o');
plot(TRAIN_FRACS,meanTestFalsePosMat, 'g-o');
xlabel('Training Fraction');
ylabel('Error rate');
legend('Test', 'false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
fname = sprintf('results_%s.fig', datestr(now, 'dd.mm.yy_HH.MM.SS'));
saveas(h, fname);
