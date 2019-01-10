function [ irl_result ] = elirltest(algorithm_params,model,mdp_model,mdp_data,...
    feature_data,true_features,taskid, time)
%ELIRLGETRESULT Summary of this function goes here
%   Detailed explanation goes here  

    algorithm_params = elirldefaultparams(algorithm_params);
    [states,actions,transitions] = size(mdp_data.sa_p);
    if algorithm_params.all_features,
        F = feature_data.splittable;
        % Add dummy feature.
        F = horzcat(F,ones(states,1));
    elseif algorithm_params.true_features,
        F = true_features;
    else
        F = eye(states);
    end;
    
    r = F*(model.L*model.S(:,taskid));
    % Return corresponding reward function.
    mdp_solve = str2func(strcat(mdp_model,'solve'));
    r = repmat(r,1,actions);
    soln = mdp_solve(mdp_data,r);
    v = soln.v;
    q = soln.q;
    p = soln.p;

    % Construct returned structure.
    irl_result = struct('r',r,'v',v,'p',p,'q',q,'r_itr',{{r}},'model_itr',{{model.L*model.S}},...
        'model_r_itr',{{r}},'p_itr',{{p}},'model_p_itr',{{p}},...
        'time',time);
end

