%%
% Create a new ELLA model
%
% inputs -
% options: a struct consisting of key value pairs
% The valid options are:
%	d (required): the dimensionality of the input data points 
%	k (required): the number of latent basis functions
%	useLogistic (default false): true if you want to do a logistic model
%		with log-loss and false for a linear model with squared loss.
%	mu (default 1): the L1 regularization constant
%	lambda (default 1): the L2 regularization constant
%	ridgeTerm (default 1): the L2 regularization constant for learning theta
%	muRatio (default Inf): the L2 regularization constant penalty for the
%			       task specific model component
%	initializeWithFirstKTasks (default false): true if you want to use the
%						   first k single task models to
%						   initialize L, false to
%						   initialize L randomly
%	lastFeatureIsABiasTerm (default false):	true if the last feature is a
%						bias term.  Bias features are
%						not regularized.
%	basisInitializationSeed: used to control the random initialization of
%				 the basis L.  This is useful for peforming
%				 controlled experiments.
%
% outputs -
% model: the created model
%
% Copyright (C) Paul Ruvolo and Eric Eaton 2013
%
% This file is part of ELLA.
%
% ELLA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ELLA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ELLA.  If not, see <http://www.gnu.org/licenses/>.
function [algorithm_params, model] = elirlcdinitmodel(algorithm_params,model,feature_data,mdp_data)
    
    algorithm_params = elirlcddefaultparams(algorithm_params);
    groupid = algorithm_params.group_id;
    [states,~,~] = size(mdp_data.sa_p);
    
    if algorithm_params.all_features
        F = feature_data.splittable;
        % Add dummy feature.
        F = horzcat(F,ones(states,1));
    elseif algorithm_params.true_features
        F = true_features;
    else
        F = eye(states);
    end
    
    if isfield(model,'isInit') && model.isInit
        if isfield(model, 'Groups') && ~ismember(groupid,model.Groups)
            model.Groups = [model.Groups; groupid];
            model.G = model.G + 1;
            model.Tg(groupid) = 0;
            model.dg(groupid) = size(F,2);
            model.Psi{groupid} = randn(model.dg(groupid), model.d);
            model.A{groupid} = zeros(model.dg(groupid) * algorithm_params.k);
            model.b{groupid} = zeros(model.dg(groupid) * algorithm_params.k, 1);
        end
        return;
    end
    
    % Count features.
    dg(groupid) = size(F,2);
    d = algorithm_params.d;
    
    Groups = groupid;
    G = 1;
    Tg(groupid) = 0;
    D = cell(sum(Tg),1);
    S = zeros(algorithm_params.k,sum(Tg));
    A = cell(1,1);
    b = cell(1,1);
    Psi = cell(1,1);
    A{groupid} = zeros(dg(groupid)*algorithm_params.k);
    b{groupid} = zeros(dg(groupid)*algorithm_params.k,1);
    Psi{groupid} = zeros(dg(groupid), d);
    theta = cell(sum(Tg),1);
    taskSpecific = cell(sum(Tg),1);
    rand('seed',algorithm_params.seed);
    randn('seed',algorithm_params.seed);
    L = randn(d,algorithm_params.k);
    
    model = struct(...
           'd',d,...
           'dg',dg,...
		   'Tg',Tg,...
           'G',G,...
           'Groups',Groups,...
		   'A',{A},...
		   'b',{b},...
		   'D',{D},...
		   'S',S,...
		   'theta',{theta},...
		   'taskSpecific',{taskSpecific},...
		   'L',L,...
           'Psi',{Psi},...
           'isInit',true);
end
