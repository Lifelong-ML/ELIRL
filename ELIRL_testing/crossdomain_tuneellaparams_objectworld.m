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
% test_result = {length(training_samples), length(percentages_trained)};

mu = 10.^(-3:2);
lambda = 10.^(-3:2);
ks = 3:10;

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

fprintf('Loaded ST model %d\n',i);

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
        % r_diff = norm(true_r - learned_r);
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

%% Train for all ks, fix others
test_result_elirl = cell(length(ks), num_perms, length(num_tasks));

cumsum_numtasks = cumsum(num_tasks);
for i = 1:length(ks)
    for j = 1:num_perms
        for k = 1 : length(num_tasks)
            fprintf('k = %d, permutation = %d\n',ks(i),j);
            idx0 = cumsum_numtasks(k) - num_tasks(k) + 1;
            idxf = idx0 + num_tasks(k) - 1;
            permutations_k = permutations(j, bsxfun(@and, permutations(j,:) >= idx0, permutations(j,:) <= idxf));
            st_model_k = st_model(permutations_k);
            permutations_k = permutations_k - idx0 + 1;
            
            tasks_elirl = struct('num_tasks', num_tasks(k), 'feature_data', {feature_data(idx0 : idxf)},...
                'example_samples', {example_samples(idx0:idxf)}, 'mdp_data', {mdp_data(idx0 : idxf)},...
                'true_feature_map', {true_feature_map(idx0 : idxf)}, 'mdp_solution', {mdp_solution(idx0 : idxf)},...
                'train_percent', 0, 'r', {r(idx0 : idxf)},'mdp_model',mdp_model);
            algorithm_params = struct('k',ks(i),...
                'mu',mu(1),...
                'lambda',lambda(1),...
                'updateLMethod','blockCoordinate',...
                'updatePsiMethod','fullByLColumn',...
                'numLPsiUpdates',1,...
                'updatePsiWithOldL',false,...
                'hess_approx','use_cov',...
                'initializeWithFirstKTasks',true);
            algorithm_params.num_tasks = num_tasks;
            % Init Model
            test_params.training_samples = training_samples;
            tasks_cd.train_percent = percentages_trained;
            [test_result_elirl{i,j,k}] = runmultitaskxval('elirl',algorithm_params,mdp_model,...
                mdp,mdp_params_all(idx0 : idxf),test_params, tasks_elirl,st_model_k,permutations_k, n_folds);
        end
    end
end

%% Find best ks

val_elirl = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirl = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirl = zeros(length(ks),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirl = zeros(length(ks),length(num_tasks));
avg_val_diff_elirl = zeros(length(ks),length(num_tasks));
avg_r_diff_elirl = zeros(length(ks),length(num_tasks));

for i = 1:length(ks)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                r_diff = test_result_elirl{i,j,k}{t}.metric_scores{2,8};
                val = test_result_elirl{i,j,k}{t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirl(i,j,k,t) = value;
                val_diff_elirl(i,j,k,t) = diff_val;
                r_diff_elirl(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirl(i,k) = mean(reshape(val_elirl(i,:,k,:),reshape_size));
        avg_val_diff_elirl(i,k) = mean(reshape(val_diff_elirl(i,:,k,:),reshape_size));
        avg_r_diff_elirl(i,k) = mean(reshape(r_diff_elirl(i,:,k,:),reshape_size));
    end
end

[~, minidx] = min(avg_r_diff_elirl);
kstar = ks(minidx);
fprintf('kstar = %d\n',kstar);

%% Train for all mus, fix others
test_result_elirl = cell(length(mu), num_perms, length(num_tasks));

cumsum_numtasks = cumsum(num_tasks);
for i = 1:length(mu)
    for j = 1:num_perms
        for k = 1 : length(num_tasks)
            fprintf('mu = %d, permutation = %d\n',mu(i),j);
            idx0 = cumsum_numtasks(k) - num_tasks(k) + 1;
            idxf = idx0 + num_tasks(k) - 1;
            permutations_k = permutations(j, bsxfun(@and, permutations(j,:) >= idx0, permutations(j,:) <= idxf));
            st_model_k = st_model(permutations_k);
            permutations_k = permutations_k - idx0 + 1;
            
            tasks_elirl = struct('num_tasks', num_tasks(k), 'feature_data', {feature_data(idx0 : idxf)},...
                'example_samples', {example_samples(idx0:idxf)}, 'mdp_data', {mdp_data(idx0 : idxf)},...
                'true_feature_map', {true_feature_map(idx0 : idxf)}, 'mdp_solution', {mdp_solution(idx0 : idxf)},...
                'train_percent', 0, 'r', {r(idx0 : idxf)},'mdp_model',mdp_model);
            algorithm_params = struct('k',kstar(k),...
                'mu',mu(i),...
                'lambda',lambda(1),...
                'updateLMethod','blockCoordinate',...
                'updatePsiMethod','fullByLColumn',...
                'numLPsiUpdates',1,...
                'updatePsiWithOldL',false,...
                'hess_approx','use_cov',...
                'initializeWithFirstKTasks',true);
            algorithm_params.num_tasks = num_tasks;
            % Init Model
            test_params.training_samples = training_samples;
            tasks_cd.train_percent = percentages_trained;
            [test_result_elirl{i,j,k}] = runmultitaskxval('elirl',algorithm_params,mdp_model,...
                mdp,mdp_params_all(idx0 : idxf),test_params, tasks_elirl,st_model_k,permutations_k, n_folds);
        end
    end
end

%% Find best mus

val_elirl = zeros(length(mu),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirl = zeros(length(mu),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirl = zeros(length(mu),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirl = zeros(length(mu),length(num_tasks));
avg_val_diff_elirl = zeros(length(mu),length(num_tasks));
avg_r_diff_elirl = zeros(length(mu),length(num_tasks));

for i = 1:length(mu)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                r_diff = test_result_elirl{i,j,k}{t}.metric_scores{2,8};
                val = test_result_elirl{i,j,k}{t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirl(i,j,k,t) = value;
                val_diff_elirl(i,j,k,t) = diff_val;
                r_diff_elirl(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirl(i,k) = mean(reshape(val_elirl(i,:,k,:),reshape_size));
        avg_val_diff_elirl(i,k) = mean(reshape(val_diff_elirl(i,:,k,:),reshape_size));
        avg_r_diff_elirl(i,k) = mean(reshape(r_diff_elirl(i,:,k,:),reshape_size));
    end
end

[~, minidx] = min(avg_r_diff_elirl);
mustar = mu(minidx);
fprintf('mustar = %d\n',mustar);

%% Train for all lambdas, fix others
test_result_elirl = cell(length(lambda), num_perms, length(num_tasks));

cumsum_numtasks = cumsum(num_tasks);
for i = 1:length(lambda)
    for j = 1:num_perms
        for k = 1 : length(num_tasks)
            fprintf('lambda = %d, permutation = %d\n',lambda(i),j);
            idx0 = cumsum_numtasks(k) - num_tasks(k) + 1;
            idxf = idx0 + num_tasks(k) - 1;
            permutations_k = permutations(j, bsxfun(@and, permutations(j,:) >= idx0, permutations(j,:) <= idxf));
            st_model_k = st_model(permutations_k);
            permutations_k = permutations_k - idx0 + 1;
            
            tasks_elirl = struct('num_tasks', num_tasks(k), 'feature_data', {feature_data(idx0 : idxf)},...
                'example_samples', {example_samples(idx0:idxf)}, 'mdp_data', {mdp_data(idx0 : idxf)},...
                'true_feature_map', {true_feature_map(idx0 : idxf)}, 'mdp_solution', {mdp_solution(idx0 : idxf)},...
                'train_percent', 0, 'r', {r(idx0 : idxf)},'mdp_model',mdp_model);
            algorithm_params = struct('k',kstar(k),...
                'mu',mustar(k),...
                'lambda',lambda(i),...
                'updateLMethod','blockCoordinate',...
                'updatePsiMethod','fullByLColumn',...
                'numLPsiUpdates',1,...
                'updatePsiWithOldL',false,...
                'hess_approx','use_cov',...
                'initializeWithFirstKTasks',true);
            algorithm_params.num_tasks = num_tasks;
            % Init Model
            test_params.training_samples = training_samples;
            tasks_cd.train_percent = percentages_trained;
            [test_result_elirl{i,j,k}] = runmultitaskxval('elirl',algorithm_params,mdp_model,...
                mdp,mdp_params_all(idx0 : idxf),test_params, tasks_elirl,st_model_k,permutations_k, n_folds);
        end
    end
end

%% Find best lambdas

val_elirl = zeros(length(lambda),num_perms,length(num_tasks),num_tasks(1));
val_diff_elirl = zeros(length(lambda),num_perms,length(num_tasks),num_tasks(1));
r_diff_elirl = zeros(length(lambda),num_perms,length(num_tasks),num_tasks(1));

avg_val_elirl = zeros(length(lambda),length(num_tasks));
avg_val_diff_elirl = zeros(length(lambda),length(num_tasks));
avg_r_diff_elirl = zeros(length(lambda),length(num_tasks));

for i = 1:length(lambda)
    for k = 1 : length(num_tasks)
        for j = 1 : num_perms
            for t = 1:num_tasks(k)
                r_diff = test_result_elirl{i,j,k}{t}.metric_scores{2,8};
                val = test_result_elirl{i,j,k}{t}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirl(i,j,k,t) = value;
                val_diff_elirl(i,j,k,t) = diff_val;
                r_diff_elirl(i,j,k,t) = r_diff;
            end
        end
        reshape_size = [num_tasks(k)*num_perms, 1];
        avg_val_elirl(i,k) = mean(reshape(val_elirl(i,:,k,:),reshape_size));
        avg_val_diff_elirl(i,k) = mean(reshape(val_diff_elirl(i,:,k,:),reshape_size));
        avg_r_diff_elirl(i,k) = mean(reshape(r_diff_elirl(i,:,k,:),reshape_size));
    end
end

[~, minidx] = min(avg_r_diff_elirl);
lambdastar = lambda(minidx);
fprintf('mustar = %d\n',mustar);

% fprintf('k = %d, mu = %d, lambda = %d', kstar, mustar, lambdastar);
kstar
mustar
lambdastar
%% Beep

beep on
beep
beep
beep
beep off

