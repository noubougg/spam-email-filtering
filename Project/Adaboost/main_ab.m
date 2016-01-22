%adaboost boost iterations
T = 40;

RUNS_PER_FRAC = 1;
TRAIN_FRACS = .1:.1:.9;
NUM_TRAIN_FRACS = length(TRAIN_FRACS);
DIRNAME ='../Data/enron1';
testErrorMat     = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
trainErrorMat    = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
testFalsePosMat  = zeros(RUNS_PER_FRAC, NUM_TRAIN_FRACS);
for iTrainFrac = 1:NUM_TRAIN_FRACS
    trainFrac = TRAIN_FRACS(iTrainFrac)
    for run=1:RUNS_PER_FRAC
        display(run);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'train',trainFrac,run-1);
        train = importdata(fname);
        fname = sprintf('%s/%s_%g_%g.txt',DIRNAME,'test', trainFrac,run-1);
        test  = importdata(fname);
        [trainErrorMat(run,iTrainFrac) , ...
            testErrorMat(run,iTrainFrac), ...
            testFalsePosMat(run,iTrainFrac) ] ...
            = adaboost(T,train,test,trainFrac,run==1);
        err = testErrorMat(run,iTrainFrac)
    end
end

meanTestErrorMat = mean(testErrorMat, 1);
meanTrainErrorMat = mean(trainErrorMat, 1);
meanTestFalsePosMat= mean(testFalsePosMat, 1);

h = figure; 
hold on;
plot(TRAIN_FRACS,meanTestErrorMat, 'r-o');
plot(TRAIN_FRACS,meanTrainErrorMat, 'b-o');
plot(TRAIN_FRACS,meanTestFalsePosMat, 'g-o');
xlabel('Training Fraction');
ylabel('Error rate');
legend('Test', 'Train', 'false pos');
txt = sprintf('Average of %d runs per training size', RUNS_PER_FRAC);
title(txt)
saveas(h, 'train_frac.fig');

