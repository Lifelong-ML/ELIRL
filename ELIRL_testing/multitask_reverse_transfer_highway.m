% Convenience script for running a single test.
addpaths;
num_tasks = 100;
training_samples = 256;
percentages_trained = [10,30,70,100];
training_sample_lengths = 16;

%Generate Training Data

% Construct MDP and features.
mdp_data = cell(num_tasks,1);
r = cell(num_tasks,1); 
feature_data = cell(num_tasks,1);
true_feature_map = cell(num_tasks,1);
mdp_solution = cell(num_tasks,1);
mdp_params_all = cell(num_tasks,1);
mdp = 'highway2';
mdp_model = 'linearmdp';
test_params = setdefaulttestparams(struct('training_samples',training_samples(end),'verbosity',2,'training_sample_lengths',training_sample_lengths));
mdp_params = struct('discount',0.9,'determinism',0.7);
rand('seed',0);
for i = 1:num_tasks
    [mdp_data{i}, r{i}, feature_data{i}, true_feature_map{i},mdp_params_all{i}] = feval(strcat(mdp,'build'),mdp_params);
    if ~isempty(test_params.true_features),
        true_feature_map{i} = test_params.true_features;
    end
    % Solve example.
    mdp_solution{i} = feval(strcat(mdp_model,'solve'),mdp_data{i},r{i});
end


% Sample example trajectories.
example_samples = {num_tasks};
for i = 1:num_tasks
    if isempty(test_params.true_examples),
        example_samples{i} = sampleexamples(mdp_model,mdp_data{i},mdp_solution{i},test_params);
    else
        example_samples{i} = test_params.true_examples;
    end
end

% Make structure of models and params for all tasks to pass to
% runmultitasktesst.m
tasks = struct('num_tasks', num_tasks, 'feature_data', {feature_data},...
    'example_samples', {example_samples}, 'mdp_data', {mdp_data},...
    'true_feature_map', {true_feature_map}, 'mdp_solution', {mdp_solution},...
    'train_percent', 100, 'r', {r},'mdp_model',mdp_model);

fprintf('Starting tests\n');
%Start Tests
mu = 10^-0;
lambda = 10^-2;

%% Maxent
% Init Model
model = struct();
test_params.training_samples = training_samples;
[test_result_maxent, st_model] = runmultitasktest_notmt_transfer('maxent',struct(),mdp_model,...
   mdp,mdp_params_all,test_params, tasks);
% tmp = load('./dumps/st_model_highway2.mat');
% tmp = load('./dumps/maxent_highway2_19-Feb-2017_132934.mat');
% st_model = tmp.st_model;
fprintf('Loaded ST model\n',i);

rand('seed', 25);
num_perms = 20;
for j = 1:num_perms   
    permutations(j,:) = randperm(num_tasks);
end

delta_err_all = zeros(num_tasks, num_perms);
delta_err_retr_lasso_all = zeros(num_tasks, num_perms);
for j = 1:num_perms
    algorithm_params = struct('k',4,'mu',mu,'lambda',lambda,'hess_approx','use_cov','initializeWithFirstKTasks',true);
    % Init Model
    test_params.training_samples = training_samples;
    [delta_err,delta_err_retr_lasso] = run_reverse_transfer_test('elirl',algorithm_params,mdp_model,...
        mdp,mdp_params_all,test_params, tasks, st_model(permutations(j,:)), permutations(j,:));
    delta_err_all(:,j) = delta_err;
    delta_err_retr_lasso_all(:,j) = delta_err_retr_lasso;
end
%% Print Results

save('reverse_xfer_highway.mat','delta_err_all','delta_err_retr_lasso_all','lambda','mu','num_tasks');

beep
