#!/usr/bin/env python3
"""
Utility functions for the distributed DistilGPT-2 system
"""

import torch
import numpy as np
import logging
from src.models.data_models import TensorData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tensor_to_dict(tensor: torch.Tensor) -> TensorData:
    """Convert tensor to serializable format"""
    return TensorData(
        data=tensor.detach().cpu().numpy().tolist(),
        shape=list(tensor.shape),
        dtype=str(tensor.dtype)
    )

def dict_to_tensor(tensor_data: TensorData, device: str) -> torch.Tensor:
    """Convert serializable format back to tensor"""
    data = np.array(tensor_data.data).reshape(tensor_data.shape)
    tensor = torch.from_numpy(data)
    
    # Convert dtype
    if "float" in tensor_data.dtype:
        tensor = tensor.float()
    elif "long" in tensor_data.dtype:
        tensor = tensor.long()
    
    return tensor.to(device)

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger"""
    return logging.getLogger(name) 