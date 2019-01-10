% Run IRL test with specified algorithm and example.
function [test_result, test_result_no_reopt] = runmultitasktest_transfer(algorithm, algorithm_params,...
    mdp_model, mdp, mdp_params, test_params, tasks, st_model,permutation)

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

irl_result = cell(tasks.num_tasks,1);
irl_result_no_reopt = cell(tasks.num_tasks,1);
% Run Multitask-IRL algorithm.
time = 0;
model = struct();

mdp_data_xfer = cell(tasks.num_tasks,1);
r_xfer = cell(tasks.num_tasks,1);
feature_data_xfer = cell(tasks.num_tasks,1);
true_feature_map_xfer = cell(tasks.num_tasks,1);
mdp_solution_xfer = cell(tasks.num_tasks,1);
rand('seed',500);
for i = 1:tasks.num_tasks
    [mdp_data_xfer{i},r_xfer{i},feature_data_xfer{i},true_feature_map_xfer{i}] = ...
        feval(strcat(mdp,'build'),mdp_params{i});
    mdp_solution_xfer{i} = feval(strcat(tasks.mdp_model,'solve'),mdp_data_xfer{i},r_xfer{i});
end
% Psiold = {0,0};
% Lold = 0;
% deltaPsi = {[],[]};
for i = 1:tasks.num_tasks
    %Change task id
    if((i/tasks.num_tasks)*100 <= tasks.train_percent)
        algorithm_params.justEncode = false;
    else
        algorithm_params.justEncode = true;
    end
    algorithm_params.task_id = i;
    if isfield(algorithm_params,'num_tasks')
        algorithm_params.group_id = sum(permutation(i) > cumsum(algorithm_params.num_tasks)) + 1;
    end
    tic;
    model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{permutation(i)},mdp_model,...
            tasks.feature_data{permutation(i)},tasks.example_samples{permutation(i)}(1:test_params.training_samples,:),...
            tasks.true_feature_map{permutation(i)},test_params.verbosity, st_model);
    
%     deltaL(i) = max(abs(Lold(:)) - model.L(:));
%     deltaPsi{algorithm_params.group_id} = [deltaPsi{algorithm_params.group_id}; max(abs(Psiold{algorithm_params.group_id}(:)) - abs(model.Psi{algorithm_params.group_id}(:)))];
%     Psiold{algorithm_params.group_id} = model.Psi{algorithm_params.group_id};
%     Lold = model.L;
    time = time + toc;
end

time = time/tasks.num_tasks;
test_result = {tasks.num_tasks};
test_result_no_reopt = {tasks.num_tasks};
for i = 1:tasks.num_tasks
    algorithm_params.task_id = i;
    if isfield(algorithm_params,'num_tasks')
        algorithm_params.group_id = sum(permutation(i) > cumsum(algorithm_params.num_tasks)) + 1;
    end
    irl_result_no_reopt{i} = feval(strcat(algorithm,'test'),algorithm_params,model,mdp_model,mdp_data_xfer{permutation(i)},...
        feature_data_xfer{permutation(i)},true_feature_map_xfer{permutation(i)},algorithm_params.task_id,time);
    test_result_no_reopt{i} = evaluateirl(irl_result_no_reopt{i},r_xfer{permutation(i)},[],mdp_data_xfer{permutation(i)},mdp_params{permutation(i)},...
        mdp_solution_xfer{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_xfer{permutation(i)},true_feature_map_xfer{permutation(i)});
    test_result_no_reopt{i}.algorithm = algorithm;
    
    % Reoptimizing S vectors
    if ((i/tasks.num_tasks)*100 <= tasks.train_percent)
        algorithm_params.justEncode = true;
        model = feval(strcat(algorithm,'train'),algorithm_params, model ,tasks.mdp_data{i},mdp_model,...
            tasks.feature_data{i},tasks.example_samples{i}(1:test_params.training_samples,:),tasks.true_feature_map{i},test_params.verbosity, st_model);
    end
    irl_result{i} = feval(strcat(algorithm,'test'),algorithm_params,model,mdp_model,mdp_data_xfer{permutation(i)},...
        feature_data_xfer{permutation(i)},true_feature_map_xfer{permutation(i)},algorithm_params.task_id,time);
    test_result{i} = evaluateirl(irl_result{i},r_xfer{permutation(i)},[],mdp_data_xfer{permutation(i)},mdp_params{permutation(i)},...
        mdp_solution_xfer{permutation(i)},mdp,mdp_model,test_params.test_models,...
        test_params.test_metrics,feature_data_xfer{permutation(i)},true_feature_map_xfer{permutation(i)});
    test_result{i}.algorithm = algorithm;
end

% time = datestr(datetime('now'));
% time(time == ' ') = '_';
% time(time == ':') = [];
% dumpfile_name = sprintf('./dumps/%s_%s_%s.mat',algorithm,mdp,time);
% save(dumpfile_name,'model', 'irl_result','test_result');