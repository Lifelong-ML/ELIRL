%%
% Encode a new task using the specified model
%
% inputs -
% model: the ELLA model
% X: the training data (data instances are rows) 
% Y: the training labels
%
% outputs -
% s: the weights over the latent basis vectors to encode the task
% theta: the optimal single task model
% D: the hessian of the loss function evaluated about theta
% taskSpecific: the task specific model component
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
function [s, theta, H, taskSpecific] = elirlcdencodetask(model, algorithm_params,mdp_data,mdp_model,...
    feature_data,example_samples,true_features,verbosity, st_model)

    % Compute the reward function with MaxEnt
    if ~isempty(fieldnames(st_model))
        theta = st_model.theta;
        if strcmp(algorithm_params.hess_approx,'eye')
            H = st_model.hesseye;
        else
            H = st_model.hesscov;
        end
    else
        [theta, H] = elirlcdmaxentrun(algorithm_params,mdp_data,mdp_model,...
            feature_data,example_samples,true_features,verbosity);
    end

    if any(isnan(theta(:))) | any(isnan(H(:)))
        theta = zeros(size(theta));
        H = zeros(size(H));
    end
   
    % use the sparse additive modeling toolbox to encode the task
    [s, taskSpecific] = elirlcdsparseencode(model, algorithm_params,theta,H);
end
