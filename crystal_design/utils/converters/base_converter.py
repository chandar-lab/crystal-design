from abc import ABC, abstractmethod

class BaseConverter(ABC):
    '''Converts objects of TypeA to TypeB'''

    @abstractmethod
    def encode(self, decoded_obj: 'TypeA') -> 'TypeB':
        pass

    @abstractmethod
    def decode(self, encoded_obj: 'TypeB') -> 'TypeA':
        pass
