% Move indicator for human agent in interactive highway2 environment.
function highway2movehuman(mdp_params,~,s,agent)

[y,lane,speed] = highway2statetocoord(s,mdp_params);
x = (speed-1)*mdp_params.lanes+lane;
set(agent,'Position',[x-0.8,y-0.8,0.6,0.6]);
