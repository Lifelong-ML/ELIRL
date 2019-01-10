function [ L, S ] = synthetic_gridworldcreatelatent( mdp_params )
%SYNTHETIC_GRIDWORLDCREATELATENT Summary of this function goes here
%   Detailed explanation goes here
    k = mdp_params.k;
    T = mdp_params.T;
    d = (mdp_params.n-1)*2 + 1;
    L = rand(d,k);
    L = L./repmat(sum(L),d,1);
    S = zeros(T,k);
    for t = 1:T
        S(t,randsample(k,ceil(k/2))) = rand(1,ceil(k/2));
    end
end

