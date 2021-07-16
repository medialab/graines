def do_or_load(path, function):
    if not os.path.exists(path):
        res = function
    
        with open(path, 'wb') as f:
            pickle.dump(res, f)

    else:
        with open(path, 'rb') as f:
            res = pickle.load(f) 
    return res