function [ episodes ] = javaepisode( example_samples, r, mdpNumber )
import burlap.behavior.singleagent.Episode
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPState
import burlap.domain.singleagent.irlToolkitMDP.state.irlToolkitMDPListState
import burlap.mdp.core.action.Action
import burlap.mdp.core.action.ActionUtils
import burlap.mdp.core.action.UniversalActionType
import burlap.mdp.core.action.SimpleAction

import java.util.ArrayList

%JAVAEPISODE Summary of this function goes here
%   Detailed explanation goes here
    
    [num_samples, horizon] = size(example_samples);
    episodes = ArrayList;
    for i = 1:num_samples
%         s = State(example_samples{i,1}(1));
%         episode = Episode(s);
        episode = Episode();
        for j = 1:horizon
            % Subtract 1 to make compatible with java indexing
            a = example_samples{i,j}(2) - 1;    
            s = example_samples{i,j}(1) - 1;
            episode.transition(SimpleAction(int2str(a)), irlToolkitMDPState(s), r(s+1));
        end
        episodes.add(episode);
    end
end

