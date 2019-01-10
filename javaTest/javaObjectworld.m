%% imports
import burlap.behavior.functionapproximation.dense.PFFeatures;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.support.DifferentiableRF;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.debugtools.DPrint;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldTerminalFunction;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain;
import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomainList;

import burlap.statehashing.simple.SimpleHashableStateFactory;

import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.environment.Environment;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.model.SampleModel;
import burlap.mdp.singleagent.oo.OOSADomain;

import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomain
import burlap.behavior.functionapproximation.dense.FromArrayFeatures
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF
import burlap.statehashing.simple.SimpleHashableStateFactory
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest
import java.util.ArrayList
import burlap.behavior.functionapproximation.dense.FromArraySAFeatures
import burlap.domain.singleagent.irlToolkitMDP.irlToolkitMDPDomainList

import burlap.behavior.singleagent.learnfromdemo.mlirl.MultipleIntentionsMLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MultipleIntentionsMLIRLRequest;

import java.util.ArrayList;
import java.util.List;
import java.lang.String;

%% Data generation
num_tasks = 30;
training_samples = 5;
percentages_trained = 100;
training_sample_lengths = 16;

%Generate Training Data

% Construct MDP and features.
mdp_data = cell(num_tasks,1);
r = cell(num_tasks,1); 
feature_data = cell(num_tasks,1);
true_feature_map = cell(num_tasks,1);
mdp_solution = cell(num_tasks,1);
mdp_params_all = cell(num_tasks,1);
mdpName = 'objectworld2';
mdp_model = 'linearmdp';
test_params = setdefaulttestparams(struct('training_samples',training_samples(end),'verbosity',2,'training_sample_lengths',training_sample_lengths));
mdp_params = struct('discount',0.9,'determinism',0.7, 'n', 8, 'c1', 3);
rand('seed',0);
for i = 1:num_tasks
    if i == 1
        [mdp_data{i}, r{i}, feature_data{i}, true_feature_map{i},mdp_params_all{i}] = feval(strcat(mdpName,'build'),mdp_params);
        color_placement = struct();
        color_placement.map1 = mdp_data{1}.map1;
        color_placement.map2 = mdp_data{1}. map2;
        color_placement.c1array = mdp_data{1}.c1array;
        color_placement.c2array = mdp_data{1}.c2array;
    else
        [mdp_data{i}, r{i}, feature_data{i}, true_feature_map{i},mdp_params_all{i}] = feval(strcat(mdpName,'build'),mdp_params, color_placement);
    end
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

% Now for java
episodes = ArrayList; 
for i = 1:num_tasks
    episodes.addAll(javaepisode(example_samples{i}, r{i}, i));
end

%% Training
mdp = irlToolkitMDPDomain(mdp_data{1}.sa_s - 1, mdp_data{1}.sa_p, r{i}(:,1), feature_data{1}.splittable);
domain = mdp.generateDomain();
featureGen = FromArrayFeatures(domain);
rf = LinearStateDifferentiableRF(featureGen, size(feature_data{1}.splittable,2), false);

numClusters = 8;
hashingFactory = SimpleHashableStateFactory();
MIRequest = MultipleIntentionsMLIRLRequest(domain, episodes, rf, numClusters, hashingFactory);
% 
numEMSteps = 20;
learningRate = 0.003;
maxMLIRLLikelihoodChange = 0.01;
maxMLIRLSteps = 50;
MI = ArrayList;
numRepetitions = 10;
for i = 1:numRepetitions
    tic;
    MI.add(MultipleIntentionsMLIRL(MIRequest, numEMSteps, learningRate, maxMLIRLLikelihoodChange, maxMLIRLSteps, training_samples));
    MI.get(i-1).performIRL();
    training_time = toc;
end

%% Evaluation

mdp_data_xfer = cell(num_tasks,1);
r_xfer = cell(num_tasks,1);
feature_data_xfer = cell(num_tasks,1);
true_feature_map_xfer = cell(num_tasks,1);
mdp_solution_xfer = cell(num_tasks,1);
rand('seed',500);
for i = 1:num_tasks
    [mdp_data_xfer{i},r_xfer{i},feature_data_xfer{i},true_feature_map_xfer{i}] = ...
        feval(strcat(mdpName,'build'),mdp_params_all{i});
    mdp_solution_xfer{i} = feval(strcat(mdp_model,'solve'),mdp_data_xfer{i},r_xfer{i});
end

rfs = ArrayList;
thetaClusters = zeros(numClusters,MI.get(0).getClusterRFs().get(0).getNumFeatures(), numRepetitions);
membershipProbabilities = zeros(numClusters, num_tasks, numRepetitions);
r_mlirl = cell(num_tasks,numRepetitions);
irl_result = cell(num_tasks,numRepetitions);
test_result = cell(num_tasks,numRepetitions);
val_mlirl = zeros(num_tasks,numRepetitions);
val_diff_mlirl = zeros(num_tasks,numRepetitions);
r_diff_mlirl = zeros(num_tasks,numRepetitions);

for rep = 1:numRepetitions
    rfs.add(MI.get(rep-1).getClusterRFs());
    for i = 1:numClusters
        for j = 1:rfs.get(rep-1).get(i-1).getNumFeatures();
            thetaClusters(i,j,rep) = rfs.get(rep-1).get(i-1).getParameter(j-1);
        end
    end

    membershipProbabilities_tmp = MI.get(rep-1).computePerClusterMLIRLWeights();
    membershipProbabilities(:,:,rep) = membershipProbabilities_tmp(:, 1:training_samples:end);
    thetaTasks = membershipProbabilities(:,:,rep)' * thetaClusters(:,:,rep);
    
    for i = 1:num_tasks
        r_mlirl{i,rep} = feature_data_xfer{i}.splittable * thetaTasks(i,:)';
        mdp_solve = str2func(strcat(mdp_model, 'solve'));
        r_tmp = repmat(r_mlirl{i,rep}, 1, 5);
        soln = mdp_solve(mdp_data_xfer{i}, 3);
        v = soln.v;
        q = soln.q;
        p = soln.p;

        irl_result{i,rep} = struct('r', r_tmp, 'v', v, 'p', p, 'q', q, 'r_itr', {{r_tmp}}, 'model_itr', {{thetaTasks(i,:)'}},...
            'model_r_itr', {{r_tmp}}, 'p_it', {{p}}, 'model_p_itr', {{p}},...
            'time',0);

        test_result{i,rep} = evaluateirl(irl_result{i,rep}, r_xfer{i}, example_samples{i}, mdp_data_xfer{i}, mdp_params_all{i}, ...
            mdp_solution_xfer{i}, mdpName, mdp_model, test_params.test_models,...
            test_params.test_metrics, feature_data_xfer{i}, true_feature_map_xfer{i});
        test_result{i,rep}.algorithm = 'mlirl';
    end

    for i = 1:num_tasks
        r_diff = test_result{i,rep}.metric_scores{2,8};
        val = test_result{i,rep}.metric_scores(2,4);
        value = val{1}(2);
        diff_val = val{1}(1);
        val_mlirl(i,rep) = value;
        val_diff_mlirl(i,rep) = diff_val;
        r_diff_mlirl(i,rep) = r_diff;
    end
end

save('javaObjectworld_results', 'thetaClusters', 'membershipProbabilities', 'test_result',...
    'val_mlirl','val_diff_mlirl','r_diff_mlirl');
