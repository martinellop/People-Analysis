from abc import ABC, abstractmethod
import torch

class PersonDescriber(ABC):
 
    @abstractmethod
    def Extract_Description(self, x:torch.tensor):
        """tensor should be shaped [k,C,w,h], where C is for color channels, w is width and h is height."""
        pass