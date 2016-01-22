function SVMStruct = svmAlg(featureVectors, labels)

% increase the maximum number of iterations for quadprog (otherwise it
% retunrs error), and turn largescale off (otherwise we get warning).
options = optimset('maxiter',2000000,'largescale','off');

% perfrom the algorithm on training set
SVMStruct = svmtrain(featureVectors, labels, 'Method', 'QP', 'Kernel_Function', 'linear', 'QuadProg_Opts', options);  % QP SMO LS
