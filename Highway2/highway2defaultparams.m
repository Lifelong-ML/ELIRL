% Fill in default parameters for the highway example.
function mdp_params = highway2defaultparams(mdp_params)

% Create default parameters.
policy_types = {'outlaw', 'lawful', 'getaway'};
default_params = struct(...
    'seed',0,...
    'length',32,...
    'lanes',3,...
    'speeds',4,...
    'num_cars',[17 0],...
    'c1',1,...
    'c2',0,...
    'continuous',0,...
    'determinism',1.0,...
    'policy_type',policy_types(randi(3)),...
    'discount',0.9);

% Set parameters.
mdp_params = filldefaultparams(mdp_params,default_params);

% Construct default reward tree.
speeds = mdp_params.speeds;
lanes = mdp_params.lanes;
c1 = mdp_params.c1;
c2 = mdp_params.c2;
frnt_start = speeds+lanes+(c1+c2)*8*6+1;
back_start = speeds+lanes+(c1+c2)*8*7+1;
% Note that distance interleaves the classes, so +0 is 1 away with c1=1, +1
% is 1 away with c1=2, +2 is 1 from c2=1, +3 is 1 from c2=2, so to get 3
% from c1=2, need +9

police_punishment = rand();
speed_rew = rand();
if strcmp(mdp_params.policy_type,'outlaw'),
    % Outlaw policy.
    r_tree = struct('type',1,'test',frnt_start+9,'total_leaves',10,...  % Test cop in front.
            'gtTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3.
                'gtTree',struct('type',0,'index',1,'mean',police_punishment*[-10,-10,-10,-10,-10]),...
                'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Going at least speed 2.
                    'ltTree',struct('type',0,'index',2,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                    'gtTree',struct('type',0,'index',3,'mean',[0,0,0,0,0]))),...
            'ltTree',struct('type',1,'test',back_start+9,'total_leaves',7,...  % Test cop behind.
                'gtTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3.
                    'gtTree',struct('type',0,'index',4,'mean',police_punishment*[-10,-10,-10,-10,-10]),...
                    'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Going at least speed 2.
                        'ltTree',struct('type',0,'index',5,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                        'gtTree',struct('type',0,'index',6,'mean',[0,0,0,0,0]))),...
                'ltTree',struct('type',1,'test',4,'total_leaves',4,... % Go as fast as possible.
                    'ltTree',struct('type',1,'test',3,'total_leaves',3,... % Don't go slow.
                        'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Don't go slow.
                            'ltTree',struct('type',0,'index',7,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                            'gtTree',struct('type',0,'index',8,'mean',[0,0,0,0,0])),...
                        'gtTree',struct('type',0,'index',9,'mean',speed_rew*[2,2,2,2,2])),...
                    'gtTree',struct('type',0,'index',10,'mean',speed_rew*[6,6,6,6,6]))));
                
elseif strcmp(mdp_params.policy_type,'lawful'),
    % Lawful policy.
    r_tree = struct('type',1,'test',speeds+lanes,'total_leaves',7,...  % In right lane.
            'gtTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3 or above.
                'gtTree',struct('type',0,'index',1,'mean',police_punishment*[-10,-10,-10,-10,-10]),...
                'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Going at least speed 2.
                    'ltTree',struct('type',0,'index',2,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                    'gtTree',struct('type',0,'index',3,'mean',[0,0,0,0,0]))),...
            'ltTree',struct('type',1,'test',4,'total_leaves',4,...  % Test speed.
                'gtTree',struct('type',0,'index',4,'mean',speed_rew*[6,6,6,6,6]),...
                'ltTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3.
                    'gtTree',struct('type',0,'index',5,'mean',speed_rew*[2,2,2,2,2]),...
                    'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Going at least speed 2.
                        'ltTree',struct('type',0,'index',6,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                        'gtTree',struct('type',0,'index',7,'mean',[0,0,0,0,0])))));
                    
elseif strcmp(mdp_params.policy_type,'getaway'),
    % Getaway car policy.
    r_tree = struct('type',1,'test',frnt_start+9,'total_leaves',10,...  % Test cop in front.
            'gtTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3.
                'ltTree',struct('type',0,'index',1,'mean',police_punishment*[-10,-10,-10,-10,-10]),...
                'gtTree',struct('type',1,'test',4,'total_leaves',2,... % Going at least speed 4.
                    'gtTree',struct('type',0,'index',2,'mean',police_punishment*[10,10,10,10,10]),...
                    'ltTree',struct('type',0,'index',3,'mean',[0,0,0,0,0]))),...
            'ltTree',struct('type',1,'test',back_start+9,'total_leaves',7,...  % Test cop behind.
                'gtTree',struct('type',1,'test',3,'total_leaves',3,... % Going at least speed 3.
                    'ltTree',struct('type',0,'index',4,'mean',police_punishment*[-10,-10,-10,-10,-10]),...
                    'gtTree',struct('type',1,'test',4,'total_leaves',2,... % Going at least speed 4.
                        'gtTree',struct('type',0,'index',5,'mean',police_punishment*[10,10,10,10,10]),...
                        'ltTree',struct('type',0,'index',6,'mean',[0,0,0,0,0]))),...
                'ltTree',struct('type',1,'test',4,'total_leaves',4,... % Go as fast as possible.
                    'ltTree',struct('type',1,'test',3,'total_leaves',3,... % Don't go slow.
                        'ltTree',struct('type',1,'test',2,'total_leaves',2,... % Don't go slow.
                            'ltTree',struct('type',0,'index',7,'mean',speed_rew*[-2,-2,-2,-2,-2]),...
                            'gtTree',struct('type',0,'index',8,'mean',[0,0,0,0,0])),...
                        'gtTree',struct('type',0,'index',9,'mean',speed_rew*[2,2,2,2,2])),...
                    'gtTree',struct('type',0,'index',10,'mean',speed_rew*[6,6,6,6,6]))));
    
end;

% Create default parameters.
default_params = struct(...
    'r_tree',r_tree);

R_SCALE = 5;
default_params.fav_lane = 1 + (rand() > 0.5) * (mdp_params.lanes - 1);
default_params.fav_speed = 2 + (rand() > 0.5) * (mdp_params.speeds - 2);
default_params.r_lane = rand()*R_SCALE;
default_params.r_speed = rand()*R_SCALE;
if default_params.fav_lane == 1 && default_params.fav_speed ==2
    default_params.corner = 1;
elseif default_params.fav_lane == 1 && default_params.fav_speed ==4
    default_params.corner = 2;
elseif default_params.fav_lane == 3 && default_params.fav_speed == 2
    default_params.corner = 3;
elseif default_params.fav_lane == 3 && default_params.fav_speed == 4
    default_params.corner = 4;
else
    fprintf('Oh no...\n');
end

 
% Set parameters.
mdp_params = filldefaultparams(mdp_params,default_params);
