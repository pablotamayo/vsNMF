def normalize_dataset_vs_reference(dataset, reference, clims = [-2, 2]):
    """ 
    Normalizes (i.e. row standardizes) a dataset using the values of the 
    corresponding rows of a second dataset (reference)

        Parameters: 
            dataset (DataFrame): The dataset to be normalized.
            reference (DataFrame): The second dataset to be used as reference to 
                 normalized the first.
            clims (list of 2): limits for the normalized values (values higher or lower 
                 wil be clipped

        Returns: 
            df: DataFrame with the normalized dataset. 
        
     """ 
    D = dataset.T.copy()
    R = reference.T.copy()
    D1 = ((D - R.mean(axis=0))/R.std(axis=0)).clip(lower = clims[0], upper = clims[1])
    df = D1.T

    return(df)
