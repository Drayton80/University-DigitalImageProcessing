from math import sqrt, pi, cos





class SomatorioCossenoDiscreto:
    def calcula_ck(self, k:int) -> float:
        if k == 0:
            # sqrt(0.5) = 0.707106781
            return 0.707106781
        else:
            return 1.0

    def calcula_fk(self, k:int, N:int):
        return k/(2*N)

    def calcula_xk(self, k:int, n:int, N:int, Xk:float):
        Ak = ((2/N)**0.5) * self.calcula_ck(k) * Xk
        fk = self.calcula_fk
        thetak = fk*pi

        return Ak*cos(2*pi*fk*n + thetak)
    
    def somatorio_xk(self, k:int, n:int, N:int, Xk:list):
        somatorio_xk = 0
        
        for k in range(0, N-1):
            somatorio_xk += self.calcula_xk(k, n, N, X[k])

        return somatorio_xk
