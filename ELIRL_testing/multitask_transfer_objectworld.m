% Convenience script for running a single test.
addpaths;
num_tasks = 100;
training_samples = 32;
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
mdp = 'objectworld2';
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
    'train_percent', 0, 'r', {r},'mdp_model',mdp_model);

fprintf('Starting tests\n');
%Start Tests

mu = 1;
lambda = 10^-2;


%% Maxent

% Init Model
model = struct();
test_params.training_samples = training_samples;
% [test_result_maxent, st_model] = runmultitasktest_notmt_transfer('maxent',struct(),mdp_model,...
%    mdp,mdp_params_all,test_params, tasks);
tmp = load('./dumps/maxent_objectworld2_14-Feb-2017_111700.mat');
test_result_maxent = tmp.test_result_maxent;
st_model = tmp.st_model;
fprintf('Loaded ST model %d\n',i);

val_maxent = zeros(num_tasks,1);
val_diff_maxent = zeros(num_tasks,1);
r_diff_maxent = zeros(num_tasks,1);

for k = 1:num_tasks
    val = test_result_maxent{k}.metric_scores(2,4);
    value = val{1}(2);
    diff_val = val{1}(1);
    r_diff = test_result_maxent{k}.metric_scores{2,8};
    val_maxent(k) = value;
    val_diff_maxent(k) = diff_val;
    r_diff_maxent(k) = r_diff; 
end
avg_val_maxent = mean(val_maxent);
err_val_maxent = std(val_maxent)/sqrt(num_tasks);
avg_val_diff_maxent = mean(val_diff_maxent);
err_val_diff_maxent = std(val_diff_maxent)/sqrt(num_tasks);
avg_r_diff_maxent = mean(r_diff_maxent);
err_r_diff_maxent = std(r_diff_maxent)/sqrt(num_tasks);

num_perms = 3;
avg_val_elirl = zeros(length(percentages_trained),num_perms);
avg_val_diff_elirl = zeros(length(percentages_trained),num_perms);
avg_r_diff_elirl = zeros(length(percentages_trained),num_perms);
err_val_elirl = zeros(length(percentages_trained),num_perms);
err_val_diff_elirl = zeros(length(percentages_trained),num_perms);
err_r_diff_elirl = zeros(length(percentages_trained),num_perms);
test_result_elirl = cell(length(percentages_trained),num_perms);
avg_val_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
avg_val_diff_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
avg_r_diff_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
err_val_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
err_val_diff_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
err_r_diff_elirl_no_reopt = zeros(length(percentages_trained),num_perms);
test_result_elirl_no_reopt = cell(length(percentages_trained),num_perms);
permutations  = zeros(num_perms,num_tasks);

rand('seed',25);
for j = 1:num_perms   
    permutations(j,:) = randperm(num_tasks);
end

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        algorithm_params = struct('k',5,'mu',mu,'lambda',lambda,'hess_approx','use_cov','initializeWithFirstKTasks',true);
        % Init Model
        test_params.training_samples = training_samples;
        tasks.train_percent = percentages_trained(i);
        [test_result{i,j}, test_result_no_reopt{i,j}] = runmultitasktest_transfer('elirl',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks,st_model(permutations(j,:)),permutations(j,:));
    end
end

%% Print Results

val_elirl = zeros(length(percentages_trained),num_tasks,num_perms);
val_diff_elirl = zeros(length(percentages_trained),num_tasks,num_perms);
r_diff_elirl = zeros(length(percentages_trained),num_tasks,num_perms);

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        for k = 1:num_tasks
            r_diff = test_result{i,j}{k}.metric_scores{2,8};
            val = test_result{i,j}{k}.metric_scores(2,4);
            value = val{1}(2);
            diff_val = val{1}(1);
            val_elirl(i,k,j) = value;
            val_diff_elirl(i,k,j) = diff_val;
            r_diff_elirl(i,k,j) = r_diff;
        end
        avg_val_elirl(i,j) = mean(val_elirl(i,:,j));
        err_val_elirl(i,j) = std(val_elirl(i,:,j))/sqrt(num_tasks);
        avg_val_diff_elirl(i,j) = mean(val_diff_elirl(i,:,j));
        err_val_diff_elirl(i,j) = std(val_diff_elirl(i,:,j))/sqrt(num_tasks);
        avg_r_diff_elirl(i,j) = mean(r_diff_elirl(i,:,j));
        err_r_diff_elirl(i,j) = std(r_diff_elirl(i,:,j))/sqrt(num_tasks);
    end  
end

val_elirl_no_reopt = zeros(length(percentages_trained),num_tasks,num_perms);
val_diff_elirl_no_reopt = zeros(length(percentages_trained),num_tasks,num_perms);
r_diff_elirl_no_reopt = zeros(length(percentages_trained),num_tasks,num_perms);

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        for k = 1:num_tasks
            r_diff = test_result_no_reopt{i,j}{k}.metric_scores{2,8};
            val = test_result_no_reopt{i,j}{k}.metric_scores(2,4);
            value = val{1}(2);
            diff_val = val{1}(1);
            val_elirl_no_reopt(i,k,j) = value;
            val_diff_elirl_no_reopt(i,k,j) = diff_val;
            r_diff_elirl_no_reopt(i,k,j) = r_diff;
        end
        avg_val_elirl_no_reopt(i,j) = mean(val_elirl_no_reopt(i,:,j));
        err_val_elirl_no_reopt(i,j) = std(val_elirl_no_reopt(i,:,j))/sqrt(num_tasks);
        avg_val_diff_elirl_no_reopt(i,j) = mean(val_diff_elirl_no_reopt(i,:,j));
        err_val_diff_elirl_no_reopt(i,j) = std(val_diff_elirl_no_reopt(i,:,j))/sqrt(num_tasks);
        avg_r_diff_elirl_no_reopt(i,j) = mean(r_diff_elirl_no_reopt(i,:,j));
        err_r_diff_elirl_no_reopt(i,j) = std(r_diff_elirl_no_reopt(i,:,j))/sqrt(num_tasks);
    end  
end

save('elirl_objectworld.mat','avg_val_elirl','err_val_elirl',...
    'avg_val_diff_elirl','err_val_diff_elirl','avg_r_diff_elirl',...
    'err_r_diff_elirl','avg_val_elirl_no_reopt','err_val_elirl_no_reopt',...
    'avg_val_diff_elirl_no_reopt','err_val_diff_elirl_no_reopt','avg_r_diff_elirl_no_reopt',...
    'err_r_diff_elirl_no_reopt','avg_val_maxent','err_val_maxent', 'avg_val_diff_maxent',...
    'err_val_diff_maxent','avg_r_diff_maxent','err_r_diff_maxent',...
    'lambda','mu','num_tasks','percentages_trained','permutations');
beep on
beep
beep
beep
beep off