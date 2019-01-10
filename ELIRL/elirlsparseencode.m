%%
% Encode a task using the current basis and the specified single task model
%
% inputs -
% model: the ELLA model
% theta: the optimal single task model
% D: the Hessian of the loss function evaluated about theta
%
% outputs -
% s: the task encoding coefficients for the current latent basis
% taskSpecific: the task specific model component
%
% Copyright (C) Paul Ruvolo and Eric Eaton 2013
%
% This file is part of ELLA.
%
% ELLA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% ELLA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with ELLA.  If not, see <http://www.gnu.org/licenses/>.
function [s taskSpecific] = elirlsparseencode(model,algorithm_params,theta,D)
Dsqrt = real(D^.5);
suppressTaskSpecific = isinf(algorithm_params.muRatio);
target = Dsqrt*theta;
muProd = algorithm_params.mu*algorithm_params.muRatio;
dictTransformed = Dsqrt*model.L;
if ~suppressTaskSpecific
	designMatInv = inv(Dsqrt'*Dsqrt+muProd*diag([ones(size(model.L,1)-1,1); 0]))*Dsqrt';
end
converged = 0;
taskSpecific = zeros(size(model.L,1),1);
taskSpecificOld = taskSpecific;
s = zeros(algorithm_params.k,1);
sOld = s;

loss = Inf;
while ~converged
	s = full(mexLasso(target - Dsqrt*taskSpecific,dictTransformed,struct('lambda',algorithm_params.mu/2)));
	% Note: need to normalize basis vectors in order to use OMP (curently not supported)
	% s = full(mexOMP(target - Dsqrt*taskSpecific, dictTransformed, struct('L',3,'mode',1)));
	if suppressTaskSpecific
		break;
	end
	taskSpecific = designMatInv*(target - dictTransformed*s);
	lossOld = loss;
	% evaluate loss
	loss = algorithm_params.mu.*norm(s,1)+sum((target - dictTransformed*s - Dsqrt*taskSpecific).^2)+muProd*sum(taskSpecific.^2);
	if lossOld-loss < 10^-3 | loss > lossOld
		converged = 1;
	end
end
end
