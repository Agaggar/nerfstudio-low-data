import numpy as np
import os

def select_indices(metadata: dict, possible_lower: int, possible_upper: int, max_size: int, size: int, training_path, seed: int = 0, save_opts: bool = True, parser_type = "blender", data_method="random"):
    rng = np.random.default_rng(seed=seed)
    if os.path.exists(training_path):
        pass
    if data_method == "uniform":
        print("actively adding data using uniform method")
        # initial batch of data is currently fixed at 100, 102, 104, 106, 108, 110
        options = rng.choice(np.arange(possible_lower, 100), size=max_size, replace=False) #, p=probs)
        curr = np.arange(100, 112)[::2]
        options = np.concatenate((curr, options))[:size]
        # options = options[:size]
    elif data_method.lower().__contains__("nbb"):
        print("actively adding data using our method - nbb")
        # initial batch of data is currently fixed at 100, 102, 104, 106, 108, 110
        curr = rng.choice(np.arange(possible_lower, possible_upper), size=max_size, replace=False) #, p=probs)
        # curr = np.arange(100, 112)[::2][:size]
        # assumes added data will start at index 116
        options = np.arange(116, 140)
        options = np.concatenate((curr, options))[:size]
    else:
        raise ValueError(f'{data_method} is not a valid way to add data. Options are: uniform and nbb')

    sorted_images = np.zeros(len(options))
    for i, x in enumerate(np.array(metadata["frames"])[options]):
        if parser_type == "blender":
            try:
                sorted_images[i] = (str.split(str.split(x['file_path'], "images/")[-1], "_")[-1])
            except:
                sorted_images[i] = (str.split(str.split(x['file_path'], "images/")[-1], "_")[-1]).split(".png")[0]
        elif parser_type == "default":
            # sorted_images[i] = (str.split(x['file_path'], "images/")[-1][:-4])
            sorted_images[i] = (str.split(str.split(x['file_path'], "images/")[-1], "frame_")[-1])[:-4]
        else:
            raise ValueError(f'Please specify the type of data, i.e. blender if synthetic, or default if real data')

    inds = np.array(sorted_images, dtype=int)
    print("data selector", inds)
    if save_opts:
        # print("saving")
        if possible_upper > 100:
            possible_upper = 99
        all_opt = np.arange(possible_lower, possible_upper)
        mask = np.isin(all_opt, inds, invert=True)
        # candidate set
        possible_choices = all_opt[mask]
        if not isinstance(training_path, str):
            training_path = str(training_path)
        directory = os.path.dirname(os.getcwd() + "/" + training_path + "/data/selected.txt")
        os.makedirs(directory, exist_ok=True)
        np.savetxt(os.getcwd() + "/" + training_path + "/data/selected.txt", inds)
        np.savetxt(os.getcwd() + "/" + training_path + "/data/candidates.txt", possible_choices)
    return inds