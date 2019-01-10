% Fill in default parameters for the objectworld example.
function mdp_params = objectworld2defaultparams(mdp_params)

% Create default parameters.
default_params = struct(...
    'seed',0,...
    'n',32,...
    'placement_prob',0.05,...
    'c1',5,...
    'c2',2,...
    'rew_dist',3,...
    'continuous',0,...
    'determinism',1.0,...
    'discount',0.9);


% Set parameters.
mdp_params = filldefaultparams(mdp_params,default_params);

% Fill in the reward function.
R_SCALE = 5;
choose = randi([2,4]);  % choose between 2 and 4 colors;
choose_rew = randsample(mdp_params.c1,min(choose,mdp_params.c1));
rc = zeros(mdp_params.c1,1);
for i = 1:mdp_params.c1
    rc(i) = R_SCALE*(3*rand()-2)*any(choose_rew == i); % penalties up to -2*R_SCALE, rewards up to R_SCALE
end

default_params.rc = rc;
step = mdp_params.c1+mdp_params.c2;
%{
r_tree = struct('type',1,'test',1+step*2,'total_leaves',3,...       % Test distance to c1 1 shape
    'gtTree',struct('type',0,'index',1,'mean',[-2,-2,-2,-2,-2]),... % Penalty for beeing close to c1 1 shape
    'ltTree',struct('type',1,'test',2+step*1,'total_leaves',2,...   % Test distance to c1 2 shape
        'gtTree',struct('type',0,'index',2,'mean',[1 1 1 1 1]),...  % Reward for being close
        'ltTree',struct('type',0,'index',3,'mean',[0 0 0 0 0])));   % Neutral reward for any other state.
%}
r_tree = struct('type',1,'test',1+step*2,'total_leaves',3,...       % Test distance to c1 1 shape
    'ltTree',struct('type',0,'index',1,'mean',[0,0,0,0,0]),... % Neutral reward for being elsewhere
    'gtTree',struct('type',1,'test',2+step*1,'total_leaves',2,...   % Test distance to c1 2 shape
        'gtTree',struct('type',0,'index',2,'mean',(3*rand()-2)*[1 1 1 1 1]),...  % Reward for being close
        'ltTree',struct('type',0,'index',3,'mean',(3*rand()-2)*[1 1 1 1 1])));   % Penalty otherwise

% Create default parameters.
default_params.r_tree = r_tree;

% Set parameters.
mdp_params = filldefaultparams(mdp_params,default_params);
