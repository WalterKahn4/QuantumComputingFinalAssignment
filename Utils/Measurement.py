class Measurement:
    def __init__(self, 
                 index: str,
                 start: str,
                 eind: str,
                 afstand: str,
                 hoogteverschil: str,
                 std_a: str,
                 std_b: str,
                 std_c: str,
                 self_made: bool = False) -> None:
        self.start          = str(start)
        self.eind           = str(eind)
        self.index          = int(index)
        self.afstand        = int(afstand)
        self.hoogteverschil = float(hoogteverschil)
        self.std_a          = float(std_a)
        self.std_b          = float(std_b)
        self.std_c          = float(std_c)