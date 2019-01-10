function elirl_model = elirltrain(algorithm_params, elirl_model,mdp_data,mdp_model,...
    feature_data,example_samples,true_features,verbosity,st_model)
%ELIRLRUN Receives a series of tasks and runs ELIRL on them in order
%   
   
    % Initialize model
    [algorithm_params, elirl_model] = elirlinitmodel(algorithm_params,elirl_model,feature_data,mdp_data);
    elirl_model = elirladdtask(elirl_model, algorithm_params,...
        mdp_data,mdp_model,feature_data,example_samples,true_features,verbosity,st_model); 
    
end

