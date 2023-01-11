from abc import ABC, abstractmethod
from typing import List, Union, Tuple

import numpy as np


class Centroid:
    """
    Class helper for generic output type
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class BaseDetector(ABC):
    """
    Base class to build a detector on, to build a child of this class please implement the detect method
    to have something standardized
    """

    @abstractmethod
    def detect(self, path: str, visualize: bool = False) -> Union[List[Centroid], np.ndarray]:
        """
        Method to detect some puzzle piece by giving a path this should
        return Coordinates or the base image with some opencv operation
        if visualize is set to True

        Parameters
        ----------
        path : str
            Path of the image
        visualize : bool, optional
            Option to visualize the predictions, by default False

        Returns
        -------
        Union[List[Tuple[int, int]], np.ndarray]
            Or List of coordinates or base image with some circles or square on it

        Raises
        ------
        NotImplementedError
            If the method is not implement
        """
        raise NotImplementedError
