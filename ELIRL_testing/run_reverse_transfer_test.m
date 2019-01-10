% Run IRL test with specified algorithm and example.
function [delta_err,delta_err_retr_lasso] = run_reverse_transfer_test(algorithm, algorithm_params,...
    mdp_model, mdp, mdp_params, test_params, tasks, st_model, permutation)

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

% Run Multitask-IRL algorithm.
model = struct();
delta_err = zeros(tasks.num_tasks,1);
mdp_data_xfer = cell(tasks.num_tasks,1);
r_xfer = cell(tasks.num_tasks,1);
feature_data_xfer = cell(tasks.num_tasks,1);
true_feature_map_tmp_xfer = cell(tasks.num_tasks,1);
mdp_solution_xfer = cell(tasks.num_tasks,1);
rand('seed',500);
for i = 1:tasks.num_tasks
    [mdp_data_xfer{i},r_xfer{i},feature_data_xfer{i},true_feature_map_tmp_xfer{i}] = ...
        feval(strcat(mdp,'build'),mdp_params{i});
    mdp_solution_xfer{i} = feval(strcat(tasks.mdp_model,'solve'),mdp_data_xfer{i},r_xfer{i});
end

for i = 1:tasks.num_tasks
    algorithm_params.task_id = i;
    model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{permutation(i)},mdp_model,...
            tasks.feature_data{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),tasks.true_feature_map{permutation(i)},test_params.verbosity, st_model);
    irl_result_tmp = feval(strcat(algorithm,'test'),algorithm_params,model,mdp_model,mdp_data_xfer{permutation(i)},...
        feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)},algorithm_params.task_id,0);
    test_result_tmp = evaluateirl(irl_result_tmp,r_xfer{permutation(i)},[],mdp_data_xfer{permutation(i)},mdp_params{permutation(i)},...
        mdp_solution_xfer{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)});
    delta_err(i) = test_result_tmp.metric_scores{2,8};
    fprintf('Train %d\n',i);
end
delta_err_retr_lasso = delta_err;

for i = 1:tasks.num_tasks
    algorithm_params.task_id = i;
    irl_result_tmp = feval(strcat(algorithm,'test'),algorithm_params,model,mdp_model,mdp_data_xfer{permutation(i)},...
        feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)},algorithm_params.task_id,0);
    test_result_tmp = evaluateirl(irl_result_tmp,r_xfer{permutation(i)},[],mdp_data_xfer{permutation(i)},mdp_params{permutation(i)},...
        mdp_solution_xfer{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)});
    delta_err(i) = delta_err(i) - test_result_tmp.metric_scores{2,8};
    algorithm_params.justEncode = true;
    model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{permutation(i)},mdp_model,...
            tasks.feature_data{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),tasks.true_feature_map{permutation(i)},test_params.verbosity, st_model);
    irl_result_tmp = feval(strcat(algorithm,'test'),algorithm_params,model,mdp_model,mdp_data_xfer{permutation(i)},...
        feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)},algorithm_params.task_id,0);
    test_result_tmp = evaluateirl(irl_result_tmp,r_xfer{permutation(i)},[],mdp_data_xfer{permutation(i)},mdp_params{permutation(i)},...
        mdp_solution_xfer{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_xfer{permutation(i)},true_feature_map_tmp_xfer{permutation(i)});
    delta_err_retr_lasso(i) = delta_err_retr_lasso(i) - test_result_tmp.metric_scores{2,8};
    fprintf('Test %d\n',i);
end
