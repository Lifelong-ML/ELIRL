%%
% Add a new task to the specified ELLA model
%
% inputs -
% model: the ELLA model
% X: the training data for the new task (instances are rows)
% Y: the trianing labels for the new task
% taskid: the task number (the first task should be 1 and the id should
%         increment from there)
% justEncode (default false): if true just encode the task with the current
% 			      basis, but don't use the new task to update
% 			      the basis itself.  If false also update the basis
% 			      weights
%
% outputs -
% model: the updated ELLA model
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
function model = elirladdtask(model, algorithm_params,mdp_data,mdp_model,...
    feature_data,example_samples,true_features,verbosity,st_model)
    
    taskid = algorithm_params.task_id;
    % encode the new task using the current latent basis
    tic;
    [model.S(:,taskid), model.theta{taskid}, model.D{taskid}, model.taskSpecific{taskid}] ...
        = elirlencodetask(model, algorithm_params,mdp_data,mdp_model,...
            feature_data,example_samples,true_features,verbosity,st_model{taskid});
    if ~algorithm_params.justEncode
        model.T = model.T+1;
        if algorithm_params.initializeWithFirstKTasks && model.T <= algorithm_params.k
            % add a new basis vector just for the newly added task
            model.L(:,model.T) = model.theta{taskid};
            model.taskSpecific{taskid} = zeros(size(model.taskSpecific{taskid}));
            model.S(:,taskid) = 0;
            model.S(model.T,taskid) = 1;
        end
        model.A = model.A + kron(model.S(:,taskid)*model.S(:,taskid)',model.D{taskid});
        residualTheta = model.theta{taskid}-model.taskSpecific{taskid};
        tmp = kron(model.S(:,taskid)',residualTheta'*model.D{taskid});
        model.b = model.b + tmp(:);
        
        model.L = reshape((model.A./model.T + algorithm_params.lambda*eye(model.d*algorithm_params.k))\model.b./model.T,size(model.L));
        % reset any elements of L that are no longer in use
        inds = find(sum(model.L.^2)<10^-10);
        for i = 1 : length(inds)
            model.L(:,inds(i)) = randn(model.d,1);
        end
    end
    time = toc;
    model.time{taskid} = time;
    model.L_hist(:,:,taskid) = model.L;
end
