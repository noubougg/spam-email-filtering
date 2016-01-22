RUNS_PER_FRAC=10;
TRAIN_FRACS = [0.1000, 0.2000, 0.3000, 0.4000];
meanTestErrorMat = [0.0915, 0.0670, 0.0578, 0.05098];
meanTestFalsePosMat = [0.0743, 0.0545, 0.0474, 0.04517];

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
