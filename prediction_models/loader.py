import json


def load_data(config):  

    # load Mapping 
    print("Loading mapping and data")
    with open(config.data_path) as data_file:    
        dico = json.load(data_file)
        
    id_to_job = dico['title_to_id']
    train_data, test_data = dico['train_data'], dico['test_data']
    
    return id_to_job, train_data, test_data
