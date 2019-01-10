% Sample example tranjectories from the state space of a given MDP.
function [example_samples, p_path, f_path,fftrans_path] = sampletrajectories(mdp_data,...
    mdp_solution,training_samples,training_sample_lengths,F)

% Allocate training samples.
N = training_samples;
T = training_sample_lengths;
example_samples = cell(N,T);
logp = zeros(N,1);
f_path = zeros(N,size(F,2));
fftrans_path = zeros(size(F,2),size(F,2),N);
% Sample trajectories.
for i=1:N,
    % Sample initial state.
    s = ceil(rand(1,1)*mdp_data.states);
    % Run sample trajectory.
    for t=1:T,
        % Add features of current state
        f_path(i,:) = f_path(i,:) + mdp_data.discount^t*F(s,:);
%         f_path(i,:) = f_path(i,:) + F(s,:);
        
        % Compute optimal action for current state.
        a = linearmdpaction(mdp_data,mdp_solution,s);
        logp(i) = logp(i) + log(mdp_solution.p(s,a));   % log-probability of action given state
        % Store example.
        example_samples{i,t} = [s;a];
        
        % Move on to next state.
        sprev = s;
        s = linearmdpstep(mdp_data,mdp_solution,s,a);
        logp(i) = logp(i) + log(sum(mdp_data.sa_p(sprev,a,mdp_data.sa_s(sprev,a,:) == s)));
            % log-probability of state transition
    end;
    fftrans_path(:,:,i) = f_path(i,:)*f_path(i,:)';
end;

p_path = exp(logp);


