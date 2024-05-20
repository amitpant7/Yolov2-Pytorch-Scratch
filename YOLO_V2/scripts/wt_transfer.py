import torch

def transfer_wts(model):
    """
    Transfers weights from a pre-trained Darknet19 model to the YOLO model.

    This function loads the pre-trained weights from a specified file and transfers
    them to the YOLO model. It matches the weights layer-by-layer for those layers
    that start with 'stage'.

    Args:
        model (torch.nn.Module): The YOLO model to which weights are transferred.

    Returns:
        torch.nn.Module: The YOLO model with transferred weights.

    Note:
        The function assumes that the pre-trained weights are stored in a file
        named 'darknet_19_state.pt' located in the './models/' directory.
    """
    # Load pre-trained weights from Darknet19 model
    darknet19_wts = torch.load('./models/darknet_19_state.pt')
    yolo_state = model.state_dict()
    
    match_keys = []

    # Match keys that start with 'stage'
    for key in yolo_state.keys():
        if key.startswith('stage'):
            match_keys.append(key)

    print('Total Layers Matched:', len(match_keys) // 6)

    # Verify before weight transfer
    print('To Verify, Before:', yolo_state[match_keys[0]].sum())
    
    with torch.no_grad():
        for des_key, src_key in zip(match_keys, darknet19_wts.keys()):
            if yolo_state[des_key].shape == darknet19_wts[src_key].shape:
                yolo_state[des_key] = darknet19_wts[src_key]
            else:
                print('Weight Transfer Failed')
                break

    # Verify after weight transfer
    print('To Verify, After:', yolo_state[match_keys[0]].sum())
    
    model.load_state_dict(yolo_state)
    print('Weight transfer complete')
    
    return model
