import numpy as np
from copy import copy, deepcopy

class Block:
    def __init__(self, data, mask = None):
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if mask is None:
            mask = np.arange(0, data.size)
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        # We do not perform indexing because numpy idexing copies stuff
        self.data = data
        self.mask = deepcopy(mask)
        self.size = mask.size
        self.parity = self.get_parity()
        self.relative_parity = None

    def get_parity(self):
        return (np.count_nonzero(self.data[self.mask]) % 2) == 1

    def get_relative_parity(self, other_parity):
        return np.logical_xor(self.parity, other_parity)

    def get_masked_data(self):
        return self.data[self.mask]

    def contains(self, index):
        return (index in self.mask)

    def flip_parities(self):
        self.parity = not self.parity
        self.relative_parity = not self.relative_parity


class Block_List:
    def __init__(self):
        self.list = []

    def __getitem__(self, index):
        return self.list[index]
    
    def append(self, argument):
        if isinstance(argument, Block):
            self.list.append(argument)
        elif isinstance(argument, list):
            self.list += argument
    
    def needs_correction(self):
        list_of_parities = [x.relative_parity for x in self.list]
        return np.any(list_of_parities)

    def shortest_problem(self):
        best_size = np.infty
        best_index = None
        for index, block in enumerate(self.list):
            if block.relative_parity == True and block.size < best_size:
                best_size = block.size
                best_index = index
        return best_index
    


