% Convenience script for running a single test.
addpaths;
test_result = runtest('maxent',struct(),'linearmdp',...
    'gridworld2',struct('c1',1,'n',8,'determinism',0.7,'seed',1,'continuous',0),...
    struct('training_sample_lengths',8,'training_samples',1,'verbosity',2));
 
% Visualize solution.
printresult(test_result);
visualize(test_result);
