% Convenience script for running a single test.
addpaths;
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


% Tuned via cross validation on 10 tasks per domain
k_elirl = [9,8,9,10,6,5,10,8,9,8,6,9,7,9,4,9,7,9,8,6,...
    9,8,9,7,3,4,7,10,7,4,3,5,4,4,4,4,7,10,7,3];
mu_elirl = [10,10,10,10,10,10,0.001,10,100,100,100,10,100,100,10,10,1,10,1,10,...
    0.001,100,0.001,1,0.001,0.1,0.001,100,0.1,100,0.001,0.001,10,0.001,100,100,10,0.001,0.001,10];
lambda_elirl = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.1,0.01,0.001,0.01,0.1,0.01,0.01,1,100,0.1,...
    [0.001,0.001,0.1,0.01,0.1,100,0.001,0.001,0.1,0.001,0.01,0.001,0.001,10,100,0.001,0.001,100,0.1,100]];

k_crossdomain = 6;
mu_1 = 1e1;
mu_2 = 1e1;
mu_3 = 1e-3;

% Make structure of models and params for all tasks to pass to
% runmultitasktesst.m
tasks_maxent = struct('num_tasks', sum(num_tasks), 'feature_data', {feature_data},...
    'example_samples', {example_samples}, 'mdp_data', {mdp_data},...
    'true_feature_map', {true_feature_map}, 'mdp_solution', {mdp_solution},...
    'train_percent', 0, 'r', {r},'mdp_model',mdp_model);


%% Maxent

% % Init Model
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

%% Run MTL training and testing
num_perms = 10;
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
permutations = zeros(num_perms,sum(num_tasks));

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

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        algorithm_params = struct('k',k_crossdomain,...
                                'mu_1',mu_1,...
                                'mu_2',mu_2,...
                                'mu_3',mu_3,...
                                'updateLMethod','blockCoordinate',...
                                'updatePsiMethod','fullByLColumn',...
                                'learningRatePsi',1e-3,...
                                'learningRateL',1e-3,...
                                'numLPsiUpdates',1,...
                                'updatePsiWithOldL',true,...
                                'hess_approx','use_cov',...
                                'initializeWithFirstKTasks',true);
        algorithm_params.num_tasks = num_tasks;
        algorithm_params.d = algorithm_params.k;
        % Init Model
        test_params.training_samples = training_samples;
        tasks_cd.train_percent = percentages_trained(i);
        [test_result_cd{i,j}, test_result_no_reopt_cd{i,j}] = runmultitasktest_transfer('elirlcd',algorithm_params,mdp_model,...
            mdp,mdp_params_all,test_params, tasks_cd,st_model(permutations(j,:)),permutations(j,:));
    end
end

%% 
% Make structure of models and params for all tasks to pass to
% runmultitasktesst.m

cumsum_numtasks = cumsum(num_tasks);
for j = 1:num_perms
    for i = 1:length(percentages_trained)
        for k = 1 : length(num_tasks)
            idx0 = cumsum_numtasks(k) - num_tasks(k) + 1;
            idxf = idx0 + num_tasks(k) - 1;
            permutations_k = permutations(j, bsxfun(@and, permutations(j,:) >= idx0, permutations(j,:) <= idxf));
            st_model_k = st_model(permutations_k);
            permutations_k = permutations_k - idx0 + 1;
            tasks_elirl = struct('num_tasks', num_tasks(k), 'feature_data', {feature_data(idx0 : idxf)},...
                'example_samples', {example_samples(idx0:idxf)}, 'mdp_data', {mdp_data(idx0 : idxf)},...
                'true_feature_map', {true_feature_map(idx0 : idxf)}, 'mdp_solution', {mdp_solution(idx0 : idxf)},...
                'train_percent', 0, 'r', {r(idx0 : idxf)},'mdp_model',mdp_model);
            algorithm_params = struct('k',k_elirl(k),...
                                    'mu',mu_elirl(k),...
                                    'lambda',lambda_elirl(k),...
                                    'hess_approx','use_cov',...
                                    'initializeWithFirstKTasks',true);
            % Init Model
            test_params.training_samples = training_samples;
            tasks_elirl.train_percent = percentages_trained(i);
            [test_result{i,j,k}, test_result_no_reopt{i,j,k}] = runmultitasktest_transfer('elirl',algorithm_params,mdp_model,...
                mdp,mdp_params_all(idx0 : idxf),test_params, tasks_elirl,st_model_k,permutations_k);
    
        end
    end
end

%% Print Results

val_elirl = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_diff_elirl = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
r_diff_elirl = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_elirlcd = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_diff_elirlcd = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
r_diff_elirlcd = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        for k = 1:length(num_tasks)
            for l = 1 : num_tasks(k)
                r_diff = test_result{i,j,k}{l}.metric_scores{2,8};
                val = test_result{i,j,k}{l}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirl(i,l,j,k) = value;
                val_diff_elirl(i,l,j,k) = diff_val;
                r_diff_elirl(i,l,j,k) = r_diff;
                
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_cd{i,j}{permutations(j,:) == idx0 + l}.metric_scores{2,8};
                val = test_result_cd{i,j}{permutations(j,:) == idx0 + l}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd(i,l,j,k) = value;
                val_diff_elirlcd(i,l,j,k) = diff_val;
                r_diff_elirlcd(i,l,j,k) = r_diff;
            end
        end
    end  
end

val_elirl_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_diff_elirl_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
r_diff_elirl_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_elirlcd_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
val_diff_elirlcd_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));
r_diff_elirlcd_no_reopt = zeros(length(percentages_trained),num_tasks(1),num_perms,length(num_tasks));

for j = 1:num_perms
    for i = 1:length(percentages_trained)
        for k = 1:length(num_tasks)
            for l = 1 : num_tasks(k)
                r_diff = test_result_no_reopt{i,j,k}{l}.metric_scores{2,8};
                val = test_result_no_reopt{i,j,k}{l}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirl_no_reopt(i,l,j,k) = value;
                val_diff_elirl_no_reopt(i,l,j,k) = diff_val;
                r_diff_elirl_no_reopt(i,l,j,k) = r_diff;
                
                idx0 = sum(num_tasks(1:k)) - num_tasks(k);
                r_diff = test_result_no_reopt_cd{i,j}{permutations(j,:) == idx0 + l}.metric_scores{2,8};
                val = test_result_no_reopt_cd{i,j}{permutations(j,:) == idx0 + l}.metric_scores(2,4);
                value = val{1}(2);
                diff_val = val{1}(1);
                val_elirlcd_no_reopt(i,l,j,k) = value;
                val_diff_elirlcd_no_reopt(i,l,j,k) = diff_val;
                r_diff_elirlcd_no_reopt(i,l,j,k) = r_diff;
            end
        end
    end  
end

%%

save('crossdomain_objectworld.mat','avg_val_maxent','err_val_maxent', ...
    'avg_val_diff_maxent','err_val_diff_maxent','avg_r_diff_maxent','err_r_diff_maxent',...
    'val_elirl', 'val_diff_elirl','r_diff_elirl', 'val_elirlcd', 'val_diff_elirlcd',...
    'r_diff_elirlcd', 'val_elirl_no_reopt', 'val_diff_elirl_no_reopt', 'r_diff_elirl_no_reopt',...
    'val_elirlcd_no_reopt', 'val_diff_elirlcd_no_reopt', 'r_diff_elirlcd_no_reopt',...
    'num_tasks','percentages_trained','permutations');
beep on
beep
beep
beep
beep off