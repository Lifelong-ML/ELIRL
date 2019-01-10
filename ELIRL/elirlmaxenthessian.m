function Hessian = elirlmaxenthessian(algorithm_params,F,mdp_solution,mdp_data,initD,example_samples)
%ELIRLMAXENTHESSIAN Summary of this function goes here
%   Detailed explanation goes here

if strcmp(algorithm_params.hess_approx,'eye')                % identity matrix
    Hessian = eye(size(F,2));
    
elseif strcmp(algorithm_params.hess_approx, 'use_cov')
    [N,~] = size(example_samples);
    if mdp_data.discount < .9999
        T = ceil(log(1e-4)/log(mdp_data.discount));
    else
        T = 100*(size(F,2));
    end
    paths = 10*N;
    [f_path] = sampletrajectories(mdp_data, mdp_solution,paths,T,F);
    Hessian = cov(f_path);
end
end

