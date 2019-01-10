% Run IRL test with specified algorithm and example.
function [test_result, st_model] = runmultitasktest_notmt_transfer(algorithm, algorithm_params,...
    mdp_model, mdp, mdp_params, test_params, tasks)

% test_result - structure that contains results of the test:
%   see evaluateirl.m
% algorithm - string specifying the IRL algorithm to use; one of:
%   firl - NIPS 2010 FIRL algorithm
%   bfirl - Bayesian FIRL algorithm
% algorithm_params - parameters of the specified algorithm:
%   FIRL:
%       seed (0) - initialization for random seed
%       iterations (10) - number of FIRL iterations to take
%       depth_step (1) - increase in depth per iteration
%       init_depth (0) - initial depth
%	BFIRL:
%       seed (0) - initialization for random seed
% mdp_model - string specifying MDP model to use for examples:
%   standardmdp - standard MDP model
% mdp - string specifying example to test on:
%   gridworld
% mdp_params - string specifying parameters for example:
%   Gridworld:
%       seed (0) - initialization for random seed
%       n (32) - number of cells along each axis
%       b (4) - size of macro cells
%       determinism (1.0) - probability of correct transition
%       discount (0.9) - temporal discount factor to use
% test_params - general parameters for the test:
%   test_models - models to test on
%   test_metrics - metrics to use during testing
%   training_samples (32) - number of example trajectories to query
%   training_sample_lengths (100) - length of each sample trajectory
%   true_features ([]) - alternative set of true features


irl_result = {tasks.num_tasks};
st_model = {tasks.num_tasks};
% Run Multitask-IRL algorithm.
time = 0;
for i = 1:tasks.num_tasks
    %Change task id
    algorithm_params.task_id = i;
    tic;
    [irl_result{i},st_model{i}] = feval(strcat(algorithm,'run'),algorithm_params,tasks.mdp_data{i},mdp_model,...
            tasks.feature_data{i},tasks.example_samples{i}(1:test_params.training_samples,:),tasks.true_feature_map{i},test_params.verbosity);
    time = time + toc;
end
time = time/tasks.num_tasks;

% Evaluate result.
test_result = {tasks.num_tasks};
rand('seed',500);
for i = 1:tasks.num_tasks
    [mdp_data_tmp,r_tmp,feature_data_tmp,true_feature_map_tmp] = ...
        feval(strcat(mdp,'build'),mdp_params{i});
    mdp_solution_tmp = feval(strcat(tasks.mdp_model,'solve'),mdp_data_tmp,r_tmp);
    irl_result_tmp = feval(strcat(algorithm,'transfer'),irl_result{i},mdp_data_tmp,mdp_model,...
    feature_data_tmp,true_feature_map_tmp,test_params.verbosity);
    test_result{i} = evaluateirl(irl_result_tmp,r_tmp,[],mdp_data_tmp,mdp_params{i},...
        mdp_solution_tmp,mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_tmp,true_feature_map_tmp);
    test_result{i}.algorithm = algorithm;
    % value = test_result{i}.value + value;
end

% time = datestr(datetime('now'));
% time(time == ' ') = '_';
% time(time == ':') = [];
% dumpfile_name = sprintf('./dumps/%s_%s_%s.mat',algorithm,mdp,time);
% save(dumpfile_name,'st_model', 'irl_result','test_result');
% value = value/num_tasks;