% Sample example tranjectories from the state space of a given MDP.
function [f_path, mu_spath] = sampletrajectories(mdp_data,...
    mdp_solution,training_samples,training_sample_lengths,F)

% Allocate training samples.
N = training_samples;
T = training_sample_lengths;
example_samples = cell(N,T);
f_path = zeros(N,size(F,2));
mu_spath = zeros(N,mdp_data.states);
% Sample trajectories.
for i=1:N,
    % Sample initial state.
    s = ceil(rand(1,1)*mdp_data.states);
    f_path(i,:) = F(s,:);
    mu_spath(i,s) = 1;
    % Run sample trajectory.
    for t=1:T,
        % Compute optimal action for current state.
        a = linearmdpaction(mdp_data,mdp_solution,s);
        
        % Move on to next state.
        s = linearmdpstep(mdp_data,mdp_solution,s,a);
        f_path(i,:) = f_path(i,:) + mdp_data.discount^t*F(s,:);
        mu_spath(i,s) = mu_spath(i,s) + mdp_data.discount^t;
    end;
end;



