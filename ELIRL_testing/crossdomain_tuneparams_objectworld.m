% addpaths;
num_tasks = 10 * ones(40,1);
training_samples = 16;
percentages_trained = 100;
training_sample_lengths = [24 : -1 : 5, 24 : -1 : 5];
gridsize = [24 : -1 : 5, 24 : -1 : 5] ;
ncolors = [3 * ones(20,1), 5 * ones(20,1)];

%Generate Training Data

% Construct MDP and features.
mdp_data = cell(sum(num_tasks),1);
r = cell(sum(num_tasks),1); 
feature_data = cell(sum(num_tasks),1);
true_feature_map = cell(sum(num_tasks),1);
mdp_solution = cell(sum(num_tasks),1);
mdp_params_all = cell(sum(num_tasks),1);
mdp = 'objectworld2';
mdp_model = 'linearmdp';
test_params = setdefaulttestparams(struct('training_samples', training_samples, 'verbosity',2));
mdp_params = struct('discount',0.9,'determinism',0.7);
rand('seed',0);
% num_cars = randi(17,num_tasks,2) + 3;

counter = 1;
for i = 1 : length(num_tasks)
    for j = 1 : num_tasks(i)
        mdp_params.n = gridsize(i);
        mdp_params.c1 = ncolors(i);
        mdp_params.c2 = 1;
        [mdp_data{counter}, r{counter}, feature_data{counter}, true_feature_map{counter},mdp_params_all{counter}] = feval(strcat(mdp,'build'),mdp_params);
        if ~isempty(test_params.true_features),
            true_feature_map{counter} = test_params.true_features;
        end
        % Solve example.
        mdp_solution{counter} = feval(strcat(mdp_model,'solve'),mdp_data{counter},r{counter});
        counter = counter + 1;
    end
    dg(i) = size(feature_data{counter-1}.splittable,2) + 1;
end


% Sample example trajectories.
example_samples = {num_tasks};
counter = 1;
for i = 1:length(num_tasks)
    test_params.training_sample_lengths = training_sample_lengths(i);
    for j = 1 : num_tasks(i)
        if isempty(test_params.true_examples),
            example_samples{counter} = sampleexamples(mdp_model,mdp_data{counter},mdp_solution{counter},test_params);
        else
            example_samples{counter} = test_params.true_examples;
        end
        counter = counter + 1;
    end
end

fprintf('Starting tests\n');
%Start Tests
mu_1 = 10.^(-3:2);
mu_2 = 10.^(-3:2);
mu_3 = 10.^(-3:2);
ks = 5:10;

% Make structure of models and params for all tasks to pass to
% runmultitasktesst.m
tasks_maxent = struct('num_tasks', sum(num_tasks), 'feature_data', {feature_data},...
    'example_samples', {example_samples}, 'mdp_data', {mdp_data},...
    'true_feature_map', {true_feature_map}, 'mdp_solution', {mdp_solution},...
    'train_percent', 0, 'r', {r},'mdp_model',mdp_model);
%% Maxent

% Init Model
model = struct();
test_params.training_samples = training_samples;
[test_result_maxent, st_model] = runmultitasktest_notmt_transfer('maxent',struct(),mdp_model,...
   mdp,mdp_params_all,test_params, tasks_maxent);
% tmp = load('./dumps/maxent_objectworld2_17-May-2018_102253.mat');
% test_result_maxent = tmp.test_result;
% st_model = tmp.st_model;
fprintf('Loaded ST model\n');

%% Compute MaxEnt results
val_maxent = zeros(num_tasks(1), length(num_tasks));
val_diff_maxent = zeros(num_tasks(1), length(num_tasks));
r_diff_maxent = zeros(num_tasks(1), length(num_tasks));

counter = 1;
for k = 1 : length(num_tasks)
    for j = 1 : num_tasks(k)
        val = test_result_maxent{counter}.metric_scores(2,4);
        value = val{1}(2);
        diff_val = val{1}(1);
        r_diff = test_result_maxent{counter}.metric_scores{2,8};
        val_maxent(j,k) = value;
        val_diff_maxent(j,k) = diff_val;
        r_diff_maxent(j,k) = r_diff;
        counter = counter + 1;
    end
end

avg_val_maxent = mean(val_maxent);
err_val_maxent = std(val_maxent) ./ sqrt(num_tasks);
avg_val_diff_maxent = mean(val_diff_maxent);
err_val_diff_maxent = std(val_diff_maxent) ./ sqrt(num_tasks);
avg_r_diff_maxent = mean(r_diff_maxent);
err_r_diff_maxent = std(r_diff_maxent) ./ sqrt(num_tasks);

%% Set up task structures

num_perms = 1;
permutations  = zeros(num_perms,sum(num_tasks));

rand('seed',25);
for j = 1:num_perms   
    permutations(j,:) = randperm(sum(num_tasks));
end

% Make structure of models and params for all tasks to pass to
% runmultitasktesst.m
tasks_cd = struct('num_tasks', sum(num_tasks), 'feature_data', {feature_data},...
    'example_samples', {example_samples}, 'mdp_data', {mdp_data},...
    'true_feature_map', {true_feature_map}, 'mdp_solution', {mdp_solution},...
    'train_percent', 0, 'r', {r},'mdp_model',mdp_model);

n_folds = 10;

% %% Train for all ks, fix others
test_result_elirlcd = cell(length(ks), num_perms);
for i = 1:length(ks)
    for j = 1:num_perms
        fprintf('k = %d, permutation = %d\n',ks(i),j);
        algorithm_params = struct('k',ks(i),...
            'mu_1',mu_1(1),...
            'mu_2',mu_2(1),...
            'mu_3',mu_3(1),...
            'updateLMethod','blockCoordinate',...
            'updatePsiMethod','fullByLColumn',...
            'numLPsiUpdates',1,...
            'updatePsiWithOldL',true,...
            'hess_approx','use_cov',...
            'initializeWithFirstKTasks',true);
        algorithm_params.num_tasks = num_tasks;
        algorithm_params.d = algorithm_params.k;
        % Init Model
        test_params.training_samples = training_samples;
        tasks_cd.train_percent = percentages_trained;
        [test_result_elirlcd{i,j}] = runmultitaskxval('elirlcd',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks_cd,st_model(permutations(j,:)),permutations(j,:), n_folds);
    end
end

%% Find best k

val_elirlcd = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirlcd = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirlcd = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirlcd = zeros(length(ks),length(num_tasks));
avg_val_diff_elirlcd = zeros(length(ks),length(num_tasks));
avg_r_diff_elirlcd = zeros(length(ks),length(num_tasks));

for i = 1:length(ks)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores{2,8};
                val = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd(i,j,k,t) = value;
                val_diff_elirlcd(i,j,k,t) = diff_val;
                r_diff_elirlcd(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirlcd(i,k) = mean(reshape(val_elirlcd(i,:,k,:),reshape_size));
        avg_val_diff_elirlcd(i,k) = mean(reshape(val_diff_elirlcd(i,:,k,:),reshape_size));
        avg_r_diff_elirlcd(i,k) = mean(reshape(r_diff_elirlcd(i,:,k,:),reshape_size));
    end
end

avg_r_diff_elirlcd = (avg_r_diff_maxent - avg_r_diff_elirlcd) ./ avg_r_diff_maxent;
[~, maxidx] = max(mean(avg_r_diff_elirlcd,2));
kstar = ks(maxidx);
fprintf('kstar = %d\n',kstar);

%% Train for all mu_1s, fix others
test_result_elirlcd = cell(length(mu_1), num_perms);
for i = 1:length(mu_1)
    for j = 1:num_perms
        fprintf('mu_1 = %d, permutation = %d\n',mu_1(i),j);
        algorithm_params = struct('k',kstar,...
            'mu_1',mu_1(i),...
            'mu_2',mu_2(1),...
            'mu_3',mu_3(1),...
            'updateLMethod','blockCoordinate',...
            'updatePsiMethod','fullByLColumn',...
            'numLPsiUpdates',1,...
            'updatePsiWithOldL',true,...
            'hess_approx','use_cov',...
            'initializeWithFirstKTasks',true);
        algorithm_params.num_tasks = num_tasks;
        algorithm_params.d = algorithm_params.k;
        % Init Model
        test_params.training_samples = training_samples;
        tasks_cd.train_percent = percentages_trained;
        [test_result_elirlcd{i,j}] = runmultitaskxval('elirlcd',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks_cd,st_model(permutations(j,:)),permutations(j,:), n_folds);
    end
end

%% Find best mu_1

val_elirlcd = zeros(length(mu_1),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirlcd = zeros(length(mu_1),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirlcd = zeros(length(mu_1),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirlcd = zeros(length(mu_1),length(num_tasks));
avg_val_diff_elirlcd = zeros(length(mu_1),length(num_tasks));
avg_r_diff_elirlcd = zeros(length(mu_1),length(num_tasks));

for i = 1:length(mu_1)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores{2,8};
                val = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd(i,j,k,t) = value;
                val_diff_elirlcd(i,j,k,t) = diff_val;
                r_diff_elirlcd(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirlcd(i,k) = mean(reshape(val_elirlcd(i,:,k,:),reshape_size));
        avg_val_diff_elirlcd(i,k) = mean(reshape(val_diff_elirlcd(i,:,k,:),reshape_size));
        avg_r_diff_elirlcd(i,k) = mean(reshape(r_diff_elirlcd(i,:,k,:),reshape_size));
    end
end

avg_r_diff_elirlcd = (avg_r_diff_maxent - avg_r_diff_elirlcd) ./ avg_r_diff_maxent;
[~, maxidx] = max(mean(avg_r_diff_elirlcd,2));
mu_1star = mu_1(maxidx);
fprintf('mu_1star = %d\n',mu_1star);

%% Train for all mu_2s, fix others
test_result_elirlcd = cell(length(mu_2), num_perms);
for i = 1:length(mu_2)
    for j = 1:num_perms
        fprintf('mu_2 = %d, permutation = %d\n',mu_2(i),j);
        algorithm_params = struct('k',kstar,...
            'mu_1',mu_1star,...
            'mu_2',mu_2(i),...
            'mu_3',mu_3(1),...
            'updateLMethod','blockCoordinate',...
            'updatePsiMethod','fullByLColumn',...
            'numLPsiUpdates',1,...
            'updatePsiWithOldL',true,...
            'hess_approx','use_cov',...
            'initializeWithFirstKTasks',true);
        algorithm_params.num_tasks = num_tasks;
        algorithm_params.d = algorithm_params.k;
        % Init Model
        test_params.training_samples = training_samples;
        tasks_cd.train_percent = percentages_trained;
        [test_result_elirlcd{i,j}] = runmultitaskxval('elirlcd',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks_cd,st_model(permutations(j,:)),permutations(j,:), n_folds);
    end
end

%% Find best mu_2

val_elirlcd = zeros(length(mu_2),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirlcd = zeros(length(mu_2),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirlcd = zeros(length(mu_2),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirlcd = zeros(length(mu_2),length(num_tasks));
avg_val_diff_elirlcd = zeros(length(mu_2),length(num_tasks));
avg_r_diff_elirlcd = zeros(length(mu_2),length(num_tasks));

for i = 1:length(mu_2)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores{2,8};
                val = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd(i,j,k,t) = value;
                val_diff_elirlcd(i,j,k,t) = diff_val;
                r_diff_elirlcd(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirlcd(i,k) = mean(reshape(val_elirlcd(i,:,k,:),reshape_size));
        avg_val_diff_elirlcd(i,k) = mean(reshape(val_diff_elirlcd(i,:,k,:),reshape_size));
        avg_r_diff_elirlcd(i,k) = mean(reshape(r_diff_elirlcd(i,:,k,:),reshape_size));
    end
end

avg_r_diff_elirlcd = (avg_r_diff_maxent - avg_r_diff_elirlcd) ./ avg_r_diff_maxent;
[~, maxidx] = max(mean(avg_r_diff_elirlcd,2));
mu_2star = mu_2(maxidx);
fprintf('mu_2star = %d\n',mu_2star);

%% Train for all mu_3s, fix others
test_result_elirlcd = cell(length(mu_3), num_perms);
for i = 1:length(mu_3)
    for j = 1:num_perms
        fprintf('mu_3 = %d, permutation = %d\n',mu_3(i),j);
        algorithm_params = struct('k',kstar,...
            'mu_1',mu_1star,...
            'mu_2',mu_2star,...
            'mu_3',mu_3(i),...
            'updateLMethod','blockCoordinate',...
            'updatePsiMethod','fullByLColumn',...
            'numLPsiUpdates',1,...
            'updatePsiWithOldL',true,...
            'hess_approx','use_cov',...
            'initializeWithFirstKTasks',true);
        algorithm_params.num_tasks = num_tasks;
        algorithm_params.d = algorithm_params.k;
        % Init Model
        test_params.training_samples = training_samples;
        tasks_cd.train_percent = percentages_trained;
        [test_result_elirlcd{i,j}] = runmultitaskxval('elirlcd',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks_cd,st_model(permutations(j,:)),permutations(j,:), n_folds);
    end
end

%% Find best mu_3

val_elirlcd = zeros(length(mu_3),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirlcd = zeros(length(mu_3),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirlcd = zeros(length(mu_3),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirlcd = zeros(length(mu_3),length(num_tasks));
avg_val_diff_elirlcd = zeros(length(mu_3),length(num_tasks));
avg_r_diff_elirlcd = zeros(length(mu_3),length(num_tasks));

for i = 1:length(mu_3)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores{2,8};
                val = test_result_elirlcd{i,j}{permutations(j,:) == idx0 + t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd(i,j,k,t) = value;
                val_diff_elirlcd(i,j,k,t) = diff_val;
                r_diff_elirlcd(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirlcd(i,k) = mean(reshape(val_elirlcd(i,:,k,:),reshape_size));
        avg_val_diff_elirlcd(i,k) = mean(reshape(val_diff_elirlcd(i,:,k,:),reshape_size));
        avg_r_diff_elirlcd(i,k) = mean(reshape(r_diff_elirlcd(i,:,k,:),reshape_size));
    end
end

avg_r_diff_elirlcd = (avg_r_diff_maxent - avg_r_diff_elirlcd) ./ avg_r_diff_maxent;
[~, maxidx] = max(mean(avg_r_diff_elirlcd,2));
mu_3star = mu_3(maxidx);
fprintf('kstar = %d\n',mu_3star);

fprintf('k = %d, mu_1 = %d, mu_2 = %d, mu_3 = %d\n', kstar, mu_1star, mu_2star, mu_3star);
%% Beep

beep on
beep
beep
beep
beep off

%% d = k
% sizes = 24 : -1 : 5, 24 : -1 : 5
% colors = 3, 5
% Update Psi With Old L
% k = 6, mu_1 = 10, mu_2 = 10, mu_3 = 1e-3
