% Utility function for coverting coordinates to state indices.
function s = highway2coordtostate(x,lane,speed,mdp_params)

s = (x-1)*mdp_params.lanes*mdp_params.speeds + (lane-1)*mdp_params.speeds + speed;
