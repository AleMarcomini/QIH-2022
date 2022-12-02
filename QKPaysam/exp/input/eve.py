from netqasm.sdk.qubit import Qubit
import random

class Eve:
    def __init__(self):
        pass
    def eavesdrop(self, qubit: Qubit):
        base = random.randint(1,2)
        #best strategy: chose basis that are equal
        qubit.rot_Y(n=base+1, d=2)
        m = qubit.measure(inplace=True)
        return base, m