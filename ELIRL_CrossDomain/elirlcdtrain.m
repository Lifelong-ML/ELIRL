function elirlcd_model = elirlcdtrain(algorithm_params, elirlcd_model,mdp_data,mdp_model,...
    feature_data,example_samples,true_features,verbosity,st_model)
%elirlcdRUN Receives a series of tasks and runs elirlcd on them in order
%   
   
    % Initialize model
    [algorithm_params, elirlcd_model] = elirlcdinitmodel(algorithm_params,elirlcd_model,feature_data,mdp_data);
    elirlcd_model = elirlcdaddtask(elirlcd_model, algorithm_params,...
        mdp_data,mdp_model,feature_data,example_samples,true_features,verbosity,st_model); 
    
end

