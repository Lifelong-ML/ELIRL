% Add necessary paths for subdirectories.

% External dependencies.
addpath Utilities
addpath Utilities/plot2svg
addpath Utilities/minFunc
addpath Utilities/mtimesx_20110223/
addpath Utilities/spams-matlab/build

% General functionality.
addpath Utilities/irl-toolkit/General
addpath Utilities/irl-toolkit/Evaluation
addpath Utilities/irl-toolkit/Testing

% MDP solvers.
addpath Utilities/irl-toolkit/StandardMDP
addpath Utilities/irl-toolkit/LinearMDP
addpath Utilities/irl-toolkit/MaxEnt

% IRL algorithms.

addpath ELIRL
addpath ELIRL_CrossDomain

% Example MDPs.
addpath Objectworld2
addpath Highway2
addpath Synthetic_gridworld

% ELIRL testing
addpath ELIRL_testing

% Java testing
addpath javaTest/
javaaddpath('./burlap.jar')
% javaaddpath('~/.m2/repository/org/apache/commons/commons-lang3/3.1/commons-lang3-3.1.jar');
