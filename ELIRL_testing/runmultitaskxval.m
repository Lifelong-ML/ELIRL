% Run IRL test with specified algorithm and example.
function test_result = runmultitaskxval(algorithm, algorithm_params,...
    mdp_model, mdp, mdp_params, test_params, tasks, st_model, permutation, n_folds)

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

num_tasks = tasks.num_tasks;
irl_result = cell(num_tasks, 1);
% Run Multitask-IRL algorithm.
idx = 1:n_folds;
partitions = reshape(repmat(idx(idx <= mod(num_tasks,n_folds)), floor(num_tasks/n_folds) + 1,1), mod(num_tasks,n_folds) * (floor(num_tasks/n_folds) + 1), 1);
partitions = [partitions; reshape(repmat(idx(idx > mod(num_tasks,n_folds)), floor(num_tasks/n_folds),1), num_tasks - (mod(num_tasks,n_folds) * (floor(num_tasks/n_folds) + 1)), 1)];

for j = 1: n_folds
    time = 0;
    algorithm_params.justEncode = false;
    model = struct();
    for i = 1:tasks.num_tasks
        %Change task id
        if partitions(i) == j
            continue
        end
        algorithm_params.task_id = i;
        if isfield(algorithm_params,'num_tasks')
            algorithm_params.group_id = sum(permutation(i) > cumsum(algorithm_params.num_tasks)) + 1;
        end
        tic;
        model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{permutation(i)},mdp_model,...
                tasks.feature_data{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),tasks.true_feature_map{permutation(i)},test_params.verbosity, st_model);
        time = time + toc;
    end
    algorithm_params.justEncode = true;
    for i = 1:tasks.num_tasks
        if partitions(i) ~= j
            continue
        end
        algorithm_params.task_id = i;
        if isfield(algorithm_params,'num_tasks')
            algorithm_params.group_id = sum(permutation(i) > cumsum(algorithm_params.num_tasks)) + 1;
        end
        tic;
        model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{permutation(i)},mdp_model,...
                tasks.feature_data{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),tasks.true_feature_map{permutation(i)},test_params.verbosity, st_model);
        time = time + toc;
        irl_result{i} = feval(strcat(algorithm,'test'),algorithm_params,model,tasks.mdp_model,tasks.mdp_data{permutation(i)},...
            tasks.feature_data{permutation(i)},tasks.true_feature_map{permutation(i)},algorithm_params.task_id,time);
    end
end


% Evaluate result.
test_result = {tasks.num_tasks};
for i = 1:tasks.num_tasks
    test_result{i} = evaluateirl(irl_result{i},tasks.r{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),tasks.mdp_data{permutation(i)},mdp_params{permutation(i)},...
        tasks.mdp_solution{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,tasks.feature_data{permutation(i)},tasks.true_feature_map{permutation(i)});
    test_result{i}.algorithm = algorithm;
    % value = test_result{i}.value + value;
end
% value = value/num_tasks;
% time = datestr(datetime('now'));
% time(time == ' ') = '_';
% time(time == ':') = [];
% dumpfile_name = sprintf('./dumps/%s_%s_%s.mat',algorithm,mdp,time);
% save(dumpfile_name,'model', 'irl_result','test_result');