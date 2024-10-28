# Выделяем бх признаки АК, измененная функция из статьи 10.1109/TCBB.2022.3233627
# убрал группу, к которой принадлежит АК, это у нас уже описывается через pharmacophore_count
import numpy as np


def residueFeature(AA) -> np.array:
    def AAcharge(AA):
        if AA in ['D','E']:
            return -1.
        elif AA in ['R','K']: # убрал His
            return 1.
        else:
            return 0.

    residueFeat = []

    AAvolume = {'A':88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, 'E':138.4, 'Q':143.8, 'G':60.1, 'H':153.2, 'I':166.7, 'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, 'S':89., 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140. }
    AAhydropathy = {'A':1.8, 'R':-4.5, 'N':-3.5, 'D': -3.5, 'C': 2.5, 'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I':4.5, 'L':3.8, 'K':-3.9, 'M':1.9, 'F':2.8, 'P':-1.6, 'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V':4.2}
    AAarea = {'A':115.,'R':225.,'D':150.,'N':160.,'C':135.,'E':190.,'Q':180.,'G':75.,'H':195.,'I':175.,'L':170.,'K':200.,'M':185.,'F':210.,'P':145.,'S':115.,'T':140.,'W':255.,'Y':230.,'V':155.}
    AAweight = {'A':89.094,'R':174.203,'N':132.119,'D':133.104,'C':121.154,'E':147.131,'Q':146.146,'G':75.067,'H':155.156,'I':131.175,'L':131.175,'K':146.189,'M':149.208,'F':165.192,'P':115.132,'S':105.093,'T':119.12,'W':204.228,'Y':181.191,'V':117.148}
    AAflexibily = {'A':'1','R':'81','N':'36','D':'18','C':'3','E':'54','Q':'108','G':'1','H':'36','I':'9','L':'9','K':'81','M':'27','F':'18','P':'2','S':'3','T':'3','W':'36','Y':'18','V':'3'}
    
    residueFeat.append(AAvolume[AA])
    residueFeat.append(AAhydropathy[AA])
    residueFeat.append(AAarea[AA])
    residueFeat.append(AAweight[AA])
    residueFeat.append(AAcharge(AA))
    residueFeat.append(AAflexibily[AA])

    if AA in ('A','G','I','L','P','V'):chemical = 0 # вот для фичи не стоит считать ∆, она категориальная !!!
    elif AA in ('R','H','K'):chemical = 1
    elif AA in ('D','E'):chemical = 2
    elif AA in ('N','Q'):chemical = 3
    elif AA in ('C','M'):chemical = 4
    elif AA in ('S','T'):chemical = 5
    elif AA in ('F','W','Y'):chemical = 6
    residueFeat.append(chemical)

    if AA in ('G','A','S'):size = 0
    elif AA in ('C','D','P','N','T'):size = 1
    elif AA in ('E','V','Q','H'):size = 2
    elif AA in ('M','I','L','K','R'):size = 3
    elif AA in ('F','Y','W'):size = 4
    residueFeat.append(size)

    if AA in ('R','W','K'):hbonds = 0
    if AA in ('A','C','G','I','L','M','F','P','V'):hbonds = 1 
    if AA in ('N','Q','S','T','H','Y'):hbonds = 3  
    if AA in ('D','E'):hbonds = 4
    residueFeat.append(hbonds)
    
    return np.array(residueFeat).astype(float)