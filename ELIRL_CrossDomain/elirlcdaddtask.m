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
function model = elirlcdaddtask(model, algorithm_params,mdp_data,mdp_model,...
    feature_data,example_samples,true_features,verbosity,st_model)

    taskid = algorithm_params.task_id;
    groupid = algorithm_params.group_id;
    
    % encode the new task using the current latent basis
    tic;
    [model.S(:,taskid), model.theta{taskid}, model.D{taskid}, model.taskSpecific{taskid}] ...
        = elirlcdencodetask(model, algorithm_params,mdp_data,mdp_model,...
            feature_data,example_samples,true_features,verbosity,st_model{taskid});
    model.D{taskid} = model.D{taskid} / (model.theta{taskid}' * model.D{taskid} * model.theta{taskid}); 
    if ~algorithm_params.justEncode
        model.Tg(groupid) = model.Tg(groupid)+1;
        if algorithm_params.initializeWithFirstKTasks && sum(model.Tg) <= algorithm_params.k
            model.taskSpecific{taskid} = zeros(size(model.taskSpecific{taskid}));
            model.S(:,taskid) = 0;
            model.S(sum(model.Tg),taskid) = 1;
        end
        model.A{groupid} = model.A{groupid} + kron(model.S(:,taskid)*model.S(:,taskid)',model.D{taskid});
        residualTheta = model.theta{taskid}-model.taskSpecific{taskid};
        tmp = kron(model.S(:,taskid)',residualTheta'*model.D{taskid});
        model.b{groupid} = model.b{groupid} + tmp(:);
        
        for i = 1 : algorithm_params.numLPsiUpdates
            model.Lprev = model.L;
            model.Psiprev = model.Psi;
            model = updatePsi(model, algorithm_params, groupid);
            model = updateL(model, algorithm_params, groupid);
            if max(abs(model.Lprev(:) - model.L(:))) / max(abs(model.Lprev(:))) < 1e-3 && max(abs(model.Psiprev{groupid}(:) - model.Psi{groupid}(:))) / max(abs(model.Psi{groupid}(:))) < 1e-3
                break
            end
        end
        model.L_hist(:,:,taskid) = model.L;
    end
    time = toc;
    model.time{taskid} = time;
end

function model = updatePsi(model, algorithm_params,groupid)
    model = feval(['updatePsi_' algorithm_params.updatePsiMethod '_'], model, algorithm_params,groupid);
end

function model = updateL(model, algorithm_params,groupid)
    model = feval(['updateL_' algorithm_params.updateLMethod '_'], model, algorithm_params, groupid);
end

function model = updatePsi_fullByLColumn_(model, algorithm_params, groupid)
    if algorithm_params.updatePsiWithOldL
        L = model.Lprev;
    else
        L = model.L;
    end
    llDss = zeros(model.dg(groupid) * model.d);
    lthetaDs = zeros(model.dg(groupid) * model.d, 1);
    for i = 1 : algorithm_params.k
        idx_i = (i-1) * model.dg(groupid) + 1 : i * model.dg(groupid);
        for j = 1 : algorithm_params.k
            idx_j = (j-1) * model.dg(groupid) + 1 : j * model.dg(groupid);
            llDss = llDss + kron(L(:,i) * L(:,j)', model.A{groupid}(idx_i,idx_j));
        end
        tmp = kron(L(:,i)', model.b{groupid}(idx_i));
        lthetaDs = lthetaDs + tmp(:);
    end

    model.Psi{groupid} = reshape((llDss / model.Tg(groupid) + algorithm_params.mu_2 * eye(model.dg(groupid) * model.d)) \ ...
        (lthetaDs / model.Tg(groupid)), size(model.Psi{groupid}));
end

function model = updatePsi_fullBySumT_(model, algorithm_params, groupid)
    if algorithm_params.updatePsiWithOldL
        L = model.Lprev;
    else
        L = model.L;
    end
    APsi = zeros(model.d * model.dg(groupid));
    bPsi = zeros(model.d * model.dg(groupid),1);
    AL = zeros(model.d * algorithm_params.k);
    bL = zeros(model.d * algorithm_params.k, 1);
    
    for t = 1:model.Tg(groupid)
        task = model.Tasks{groupid};
        Lst = L * model.S(:,task);
        APsi = APsi + kron(Lst * Lst', model.D{task});% ./...
%                         (obj.Tg(groupId));
        tmp = kron(Lst', (model.theta{task}'*model.D{task}));
        bPsi = bPsi + tmp(:);% / obj.Tg(groupId);
    end
%             fprintf('Max diag Psi : %d \t Max Psi : %d \t Max reg : %d \n',max(diag(obj.APsi{groupId})),max(obj.APsi{groupId}(:)),obj.mu_2);
    model.Psi{groupid} = reshape((APsi + algorithm_params.mu_2*eye(model.d*model.dg(groupid))) \ bPsi{groupid}, size(model.Psi{groupid}));
end

function model = updatePsi_gradientStepByLColumn_(model, algorithm_params, groupid)
    if algorithm_params.updatePsiWithOldL
        L = model.Lprev;
    else
        L = model.L;
    end
    sumvar = zeros(model.dg(groupid), model.d);
    for i = 1:algorithm_params.k
        idx_i = (i-1)*model.dg(groupid)+1 : i*model.dg(groupid);
        tmp = model.b{groupid}(idx_i);
        sumvar = sumvar + model.learningRatePsi(groupid) * (-2 * tmp(:) * model.L(:,i)');
        tsumvar = zeros(model.dg(groupid), model.d);
        for j = 1:algorithm_params.k
            idx_j = (j-1)*model.dg(groupid)+1 : j*model.dg(groupid);
            tsumvar = tsumvar + model.learningRatePsi(groupid) * ...
                + 2 * model.A{groupid}(idx_i,idx_j) * model.Psi{groupid} * L(:,i) * L(:,j)' ...
                / numLabels;
        end
        sumvar = sumvar + tsumvar;
    end
    model.Psi{groupid} = (model.Psi{groupid} - 1 ./ model.Tg(groupid) * sumvar - algorithm_params.learningRatePsi * 2 * algorithm_params.mu_2 * model.Psi{groupid});
end

function model = updatePsi_gradientStepBySumT_(model, algorithm_params, groupid)
    if algorithm_params.updatePsiWithOldL
        L = model.Lprev;
    else
        L = model.L;
    end
    
    TasksForGroupg = model.Tasks{groupid};
    sumvar = zeros(model.dg(groupid), mode.d);
    
    for t = 1:model.Tg(groupid)
        tsumvar = model.learningRatePsi(groupid) * (-2 * model.D{TasksForGroupg(t)} * model.theta{TasksForGroupg(t)} * model.S{TasksForGroupg(t)}' * L' ...
            + 2 * model.D{TasksForGroupg(t)} * model.Psi{groupid} * L * model.S{TasksForGroupg(t)} * (L * model.S{TasksForGroupg(t)})');
        sumvar = sumvar + tsumvar;
    end

    model.Psi{groupid} = (model.Psi{groupid} - 1 ./ model.Tg(groupid) * sumvar - algorithm_params.learningRatePsi * 2 * algorithm_params.mu_2 * model.Psi{groupid});
end

function model = updateL_blockCoordinate_(model, algorithm_params, ~)
    converged = false;
    iteration = 1;
    while ~converged
        Lold = model.L;
        for j = 1 : algorithm_params.k
            PsiDPsiss = zeros(model.d);
            Psib = zeros(model.d, 1);
            PsiDPsilss = zeros(model.d, 1);
            for g = 1 : model.G
                gid = model.Groups(g);
                idx_j = (j-1) * model.dg(gid) + 1 : j * model.dg(gid);

                PsiDPsiss = PsiDPsiss + 1 / model.Tg(gid) * model.Psi{gid}' * model.A{gid}(idx_j, idx_j) * model.Psi{gid};
                Psib = Psib + 1 / model.Tg(gid) * model.Psi{gid}' * model.b{gid}(idx_j);
                A_tmp = model.A{gid}(:,idx_j);
                A_tmp = reshape(A_tmp', model.dg(gid), model.dg(gid), algorithm_params.k);
                DPsilss = mtimesx(A_tmp, reshape(model.Psi{gid} * model.L, model.dg(gid), 1, algorithm_params.k));
                sumDPsilss = sum(DPsilss,3) - DPsilss(:,:,j);
                PsiDPsilss = PsiDPsilss + 1 / model.Tg(gid) * model.Psi{gid}' * sumDPsilss;
            end
            uj = (PsiDPsiss + algorithm_params.mu_3 * eye(model.d)) \  (Psib - PsiDPsilss);
            model.L(:,j) = uj;
        end
        converged = max(abs(model.L(:) - Lold(:)) / max(abs(Lold(:)))) < 1e-3 || iteration >= 100;
        iteration = iteration + 1;
    end
    fprintf('%d iterations for L convergence\n',iteration-1);

    inds = find(sum(model.L.^2) < 10^-10);
    for i = 1 : length(inds)
        model.L(:,inds(i)) = randn(model.d,1);
    end  
end

function model = updateL_fullBySumT_(model, algorithm_params, ~)
    AL = zeros(model.d * algorithm_params.k);
    bl = zeros(model.d * algorithm_params.k, 1);
    for g = 1:model.G
        group = model.Groups(g);
        for t = 1:length(model.Tasks{group})
            task = model.Tasks{group}(t);
            AL = AL + kron(model.S{task}*model.S{task}', model.Psi{group}'*model.D{task}*model.Psi{group});% ./...
%                             (obj.Tg(group));

            tmp = kron(model.S{task}', (model.theta{task}'*model.D{task}*model.Psi{group}));

            model.bL = model.bL + tmp(:);% / obj.Tg(group);
        end
    end
    model.L = reshape((model.AL + algorithm_params.mu_3*eye(model.d*algorithm_params.k)) \ model.bL, size(model.L));
end

function model = updateL_gradientStepByLColumn_(model, algorithm_params, groupid)
    for i = 1:algorithm_params.k
        sumAllOne = 0;
        for g = 1:model.G % Outer Summation over Groups
            group = model.Groups(g);
            idx_i = (i-1)*model.dg(group)+1 : i*model.dg(group);
            tsum = zeros(model.dg(group),1);
            for j = 1:model.k
                idx_j = (j-1)*model.dg(group)+1 : j*model.dg(group);
                tsum = tsum + model.A{group}(idx_i,idx_j) * model.Psi{group} * model.L(:,j);
            end
            tmp = model.b{group}(idx_i);
            prodOne = -2 * model.Psi{group}' * tmp(:);
            prodTwo = 2 * model.Psi{group}' * tsum;
            sumG = (prodOne + prodTwo) / numLabels;
            sumAllOne = sumAllOne + model.learningRateL(group) * sumG ./ model.Tg(group);  % JORGE: Double check this, seemed wrong in previous implementation
        end
        model.L(:,i) = model.L(:,i) - sumAllOne - model.learningRateL(groupId) * 2 * algorithm_params.mu_3 * model.L(:,i);
    end
end

function model = updateL_gradientStepBySumT_(model, algorithm_params, groupid)
    sumAllOne = 0;
    for g = 1:model.G % Outer Summation over Groups
        group = model.Groups(g);
        TasksforGroupz = model.Tasks{group};
        sumG = 0; % Resetting SumG
        for t = 1:numel(TasksforGroupz) % number of rows determines how many tasks are in group z (i.e., inner summation)
            prodOne = -2 * model.Psi{group}' * model.D{TasksforGroupz(t)} * model.theta{TasksforGroupz(t)} * model.S{TasksforGroupz(t)}';
            prodTwo = 2 * model.Psi{group}' * model.D{TasksforGroupz(t)} * model.Psi{group} * model.L * model.S{TasksforGroupz(t)} * model.S{TasksforGroupz(t)}';
            sumG = sumG + (prodOne + prodTwo);
        end
        sumAllOne = sumAllOne + algorithm_params.learningRateL(group) * sumG ./ model.Tg(group);  % JORGE: Double check this, seemed wrong in previous implementation
    end

    model.L = (model.L - sumAllOne - algorithm_params.learningRateL(groupId) * 2 * algorithm_params.mu_3 * model.L);   % JORGE: removed 1/obj.G multiplying sumAllOne
end