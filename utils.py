import pprint
import torch
  
def p_print(text): 
  pp = pprint.PrettyPrinter(indent=4)
  pp.pprint(text)

def get_device_info():
  # get index of currently selected device
    try:
        device_index = torch.cuda.current_device() # returns 0 in my case
    except:
        device_index = None

    # get number of GPUs available
    no_gpus = torch.cuda.device_count() # returns 1 in my case

    # get the name of the device
    try:
        device_name = torch.cuda.get_device_name(0) # good old Tesla K80
    except:
        device_name = None
    print(f"device index: {device_index}")
    print(f"no gpus: {no_gpus}")
    print(f"device name: {device_name}")
