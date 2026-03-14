import math
import numpy as np

" 仿真参数，主要涉及信道建模 "

Bmax = 1
zone = 1000
height = 100
LOS = 1
NLOS = 20
F = 2 * 10**9
C = 3 * 10**8
aa = 9.61
bb = 0.16
bandwidth = 5000000
noise = 10 ** (-17.3) / 1000

" 五个簇头节点位置 "
x1 = 400
y1 = 400
x2 = 450
y2 = 450
x3 = 340
y3 = 410
x4 = 360
y4 = 650
x5 = 420
y5 = 720
Sensor1_Position = [x1, y1, 0]
Sensor2_Position = [x2, y2, 0]
Sensor3_Position = [x3, y3, 0]
Sensor4_Position = [x4, y4, 0]
Sensor5_Position = [x5, y5, 0]

" Initialization "

EH1 = []
EH2 = []
EH3 = []
EH4 = []
EH5 = []
DG1 = []
DG2 = []
DG3 = []
DG4 = []
DG5 = []

# Energy Harvesting : 能量收集伪随机序列，50个时隙
# Data Generation：   簇头节点待传输数据量的伪随机序列
for cnt in range(5):
    EH1 = EH1 + [0.2, 0.3, 0.5, 0.3, 0.6, 0.8, 0.3, 0.7, 0.3, 0.5]
    EH2 = EH2 + [0.4, 0.5, 0.3, 0.8, 0.6, 0.4, 0.6, 0.3, 0.4, 0.6]
    EH3 = EH3 + [0.3, 0.5, 0.2, 0.1, 0.3, 0.4, 0.6, 0, 0.4, 0.3]
    EH4 = EH4 + [0.5, 0.3, 0.5, 0.3, 0.6, 0.8, 0.3, 0.7, 0.3, 0.5]
    EH5 = EH5 + [0.2, 0.5, 0.3, 0.8, 0.6, 0.4, 0.6, 0.3, 0.4, 0.6]

    DG1 = DG1 + [6, 6, 6, 0 ,0, 6, 0, 4, 0 ,0]
    DG2 = DG2 + [0, 0, 2, 6, 6, 0, 0, 6, 6, 2]
    DG3 = DG3 + [6, 2, 0, 6, 0, 6, 0, 0, 6, 6]
    DG4 = DG4 + [0, 2, 4, 0, 6, 0, 6, 0, 0, 6]
    DG5 = DG5 + [2, 6, 4, 2, 0, 6, 6, 6, 6, 4]

class enva(object):
    def __init__(self):
        self.reward = 0

    # 初始化状态
    def reset(self):
        # self.rate = 0
        B1state = 0.5
        B2state = 0.5
        B3state = 0.5
        B4state = 0.5
        B5state = 0.5

        E1harv = EH1[0]
        E2harv = EH2[0]
        E3harv = EH3[0]
        E4harv = EH4[0]
        E5harv = EH5[0]

        D1Gener = DG1[0]
        D2Gener = DG2[0]
        D3Gener = DG3[0]
        D4Gener = DG4[0]
        D5Gener = DG5[0]

        C11 = 0
        C12 = 0
        C13 = 0
        C14 = 0
        C15 = 0
        C21 = 0
        C22 = 0
        C23 = 0
        C24 = 0
        C25 = 0

        return np.array([B1state,B2state,B3state,B4state,B5state,E1harv,E2harv,E3harv,E4harv,E5harv,D1Gener,D2Gener,D3Gener,D4Gener,D5Gener,C11,C12,C13,C14,C15,C21,C22,C23,C24,C25])

    # 执行更新
    def step(self, s, a, h, u1x, u1y, u2x, u2y, flag):
        done = True
        B1state,B2state,B3state,B4state,B5state,E1harv,E2harv,E3harv,E4harv,E5harv,D1Gener,D2Gener,D3Gener,D4Gener,D5Gener,C11,C12,C13,C14,C15,C21,C22,C23,C24,C25 = s
        B1state = max(min(B1state + E1harv, Bmax), 0)
        B2state = max(min(B2state + E2harv, Bmax), 0)
        B3state = max(min(B3state + E3harv, Bmax), 0)
        B4state = max(min(B4state + E4harv, Bmax), 0)
        B5state = max(min(B5state + E5harv, Bmax), 0)

        # a[0]动作为无人机与簇之间服务配对，5 对 2 选择排列组合：30种
        # 例如：a[0] == 0 时，无人机 1 选择了服务簇 1 、2，无人机 2 选择服务簇 3,4
        # a[1]-a[4]为功率分配动作，a[5]、a[6]为两个无人机带宽分配比例

        if int(a[0]) == 0:
            P1 = a[1]
            P2 = a[2]
            P3 = a[3]
            P4 = a[4]
            P5 = 0
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * (1 - a[5])
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * (1 - a[6])
            BW5 = 0

        elif int(a[0]) == 1:
            P1 = a[1]
            P2 = a[2]
            P3 = a[3]
            P4 = 0
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * (1 - a[5])
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * 0
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 2:
            P1 = a[1]
            P2 = a[2]
            P3 = 0
            P4 = a[3]
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * (1 - a[5])
            BW3 = 0
            BW4 = bandwidth * a[6]
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 3:
            P1 = a[1]
            P2 = a[3]
            P3 = a[2]
            P4 = a[4]
            P5 = 0
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * (1 - a[5])
            BW4 = bandwidth * (1 - a[6])
            BW5 = 0

        elif int(a[0]) == 4:
            P1 = a[1]
            P2 = a[3]
            P3 = a[2]
            P4 = 0
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * (1 - a[5])
            BW4 = 0
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 5:
            P1 = a[1]
            P2 = 0
            P3 = a[2]
            P4 = a[3]
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = 0
            BW3 = bandwidth * (1 - a[5])
            BW4 = bandwidth * a[6]
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 6:
            P1 = a[1]
            P2 = a[3]
            P3 = a[4]
            P4 = a[2]
            P5 = 0
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * (1 - a[6])
            BW4 = bandwidth * (1 - a[5])
            BW5 = 0

        elif int(a[0]) == 7:
            P1 = a[1]
            P2 = a[3]
            P3 = 0
            P4 = a[2]
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = 0
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 8:
            P1 = a[1]
            P2 = 0
            P3 = a[3]
            P4 = a[2]
            P5 = a[4]
            BW1 = bandwidth * a[5]
            BW2 = 0
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])


        elif int(a[0]) == 9:
            P1 = a[1]
            P2 = a[3]
            P3 = a[4]
            P4 = 0
            P5 = a[2]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * (1 - a[6])
            BW4 = 0
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 10:
            P1 = a[1]
            P2 = a[3]
            P3 = 0
            P4 = a[4]
            P5 = a[2]
            BW1 = bandwidth * a[5]
            BW2 = bandwidth * a[6]
            BW3 = 0
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 11:
            P1 = a[1]
            P2 = 0
            P3 = a[3]
            P4 = a[4]
            P5 = a[2]
            BW1 = bandwidth * a[5]
            BW2 = 0
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 12:
            P1 = a[3]
            P2 = a[1]
            P3 = a[2]
            P4 = a[4]
            P5 = 0
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * (1 - a[5])
            BW4 = bandwidth * (1 - a[6])
            BW5 = 0

        elif int(a[0]) == 13:
            P1 = a[3]
            P2 = a[1]
            P3 = a[2]
            P4 = 0
            P5 = a[4]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * (1 - a[5])
            BW4 = 0
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 14:
            P1 = 0
            P2 = a[1]
            P3 = a[2]
            P4 = a[3]
            P5 = a[4]
            BW1 = 0
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * (1 - a[5])
            BW4 = bandwidth * a[6]
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 15:
            P1 = a[3]
            P2 = a[1]
            P3 = a[4]
            P4 = a[2]
            P5 = 0
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * (1 - a[6])
            BW4 = bandwidth * (1 - a[5])
            BW5 = 0

        elif int(a[0]) == 16:
            P1 = a[3]
            P2 = a[1]
            P3 = 0
            P4 = a[2]
            P5 = a[4]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = 0
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 17:
            P1 = 0
            P2 = a[1]
            P3 = a[3]
            P4 = a[2]
            P5 = a[4]
            BW1 = 0
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 18:
            P1 = a[3]
            P2 = a[1]
            P3 = a[4]
            P4 = 0
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * (1 - a[6])
            BW4 = 0
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 19:
            P1 = a[3]
            P2 = a[1]
            P3 = 0
            P4 = a[4]
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * a[5]
            BW3 = 0
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 20:
            P1 = 0
            P2 = a[1]
            P3 = a[3]
            P4 = a[4]
            P5 = a[2]
            BW1 = 0
            BW2 = bandwidth * a[5]
            BW3 = bandwidth * a[6]
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])


        elif int(a[0]) == 21:
            P1 = a[3]
            P2 = a[4]
            P3 = a[1]
            P4 = a[2]
            P5 = 0
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * (1 - a[6])
            BW3 = bandwidth * a[5]
            BW4 = bandwidth * (1 - a[5])
            BW5 = 0

        elif int(a[0]) == 22:
            P1 = a[3]
            P2 = 0
            P3 = a[1]
            P4 = a[2]
            P5 = a[4]
            BW1 = bandwidth * a[6]
            BW2 = 0
            BW3 = bandwidth * a[5]
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 23:
            P1 = 0
            P2 = a[3]
            P3 = a[1]
            P4 = a[2]
            P5 = a[4]
            BW1 = 0
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * a[5]
            BW4 = bandwidth * (1 - a[5])
            BW5 = bandwidth * (1 - a[6])

        elif int(a[0]) == 24:
            P1 = a[3]
            P2 = a[4]
            P3 = a[1]
            P4 = 0
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * (1 - a[6])
            BW3 = bandwidth * a[5]
            BW4 = 0
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 25:
            P1 = a[3]
            P2 = 0
            P3 = a[1]
            P4 = a[4]
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = 0
            BW3 = bandwidth * a[5]
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 26:
            P1 = 0
            P2 = a[3]
            P3 = a[1]
            P4 = a[4]
            P5 = a[2]
            BW1 = 0
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * a[5]
            BW4 = bandwidth * (1 - a[6])
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 27:
            P1 = a[3]
            P2 = a[4]
            P3 = 0
            P4 = a[1]
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = bandwidth * (1 - a[6])
            BW3 = 0
            BW4 = bandwidth * a[5]
            BW5 = bandwidth * (1 - a[5])

        elif int(a[0]) == 28:
            P1 = a[3]
            P2 = 0
            P3 = a[4]
            P4 = a[1]
            P5 = a[2]
            BW1 = bandwidth * a[6]
            BW2 = 0
            BW3 = bandwidth * (1 - a[6])
            BW4 = bandwidth * a[5]
            BW5 = bandwidth * (1 - a[5])


        elif int(a[0]) == 29:
            P1 = 0
            P2 = a[3]
            P3 = a[4]
            P4 = a[1]
            P5 = a[2]
            BW1 = 0
            BW2 = bandwidth * a[6]
            BW3 = bandwidth * (1 - a[6])
            BW4 = bandwidth * a[5]
            BW5 = bandwidth * (1 - a[5])

        else:
            print(" Warning ! ")

        # 上传功率最大不超过电池电量

        P1x = np.clip(P1, 0, B1state)
        P2x = np.clip(P2, 0, B2state)
        P3x = np.clip(P3, 0, B3state)
        P4x = np.clip(P4, 0, B4state)
        P5x = np.clip(P5, 0, B5state)

        d11 = np.sqrt(pow((u1x - x1), 2) + pow((u1y - y1), 2) + pow(height, 2))
        d12 = np.sqrt(pow((u1x - x2), 2) + pow((u1y - y2), 2) + pow(height, 2))
        d13 = np.sqrt(pow((u1x - x3), 2) + pow((u1y - y3), 2) + pow(height, 2))
        d14 = np.sqrt(pow((u1x - x4), 2) + pow((u1y - y4), 2) + pow(height, 2))
        d15 = np.sqrt(pow((u1x - x5), 2) + pow((u1y - y5), 2) + pow(height, 2))

        d21 = np.sqrt(pow((u2x - x1), 2) + pow((u2y - y1), 2) + pow(height, 2))
        d22 = np.sqrt(pow((u2x - x2), 2) + pow((u2y - y2), 2) + pow(height, 2))
        d23 = np.sqrt(pow((u2x - x3), 2) + pow((u2y - y3), 2) + pow(height, 2))
        d24 = np.sqrt(pow((u2x - x4), 2) + pow((u2y - y4), 2) + pow(height, 2))
        d25 = np.sqrt(pow((u2x - x5), 2) + pow((u2y - y5), 2) + pow(height, 2))

        theta11 = math.asin(height / d11)
        theta12 = math.asin(height / d12)
        theta13 = math.asin(height / d13)
        theta14 = math.asin(height / d14)
        theta15 = math.asin(height / d15)

        theta21 = math.asin(height / d21)
        theta22 = math.asin(height / d22)
        theta23 = math.asin(height / d23)
        theta24 = math.asin(height / d24)
        theta25 = math.asin(height / d25)

        # 计算传输速率，香农公式

        if P1 == 0 or BW1 ==0 :
            rate1 = 0
        elif P1 ==a[1] or  P1 ==a[2]:
            pathloss11 = (LOS - NLOS) /(1 + aa * math.exp(-bb * (theta11 - aa)))
            pathloss11 = pathloss11 + 20 * math.log10(4 * math.pi * F * d11 / C) + NLOS
            C11 = 10 ** (-pathloss11 / 10)
            rate1 = math.log10(1 + P1x * C11 / (noise * BW1)) * BW1
        elif P1 ==a[3] or  P1 ==a[4]:
            pathloss21 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta21 - aa)))
            pathloss21 = pathloss21 + 20 * math.log10(4 * math.pi * F * d21 / C) + NLOS
            C21 = 10 ** (-pathloss21 / 10)
            rate1 = math.log10(1 + P1x * C21 / (noise * BW1)) * BW1
        else:
            print(' #### warning ####')

        if P2 == 0 or BW2 ==0 :
            rate2 = 0
        elif P2 == a[1] or P2 == a[2]:
            pathloss12 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta12 - aa)))
            pathloss12 = pathloss12 + 20 * math.log10(4 * math.pi * F * d12 / C) + NLOS
            C12 = 10 ** (-pathloss12 / 10)
            rate2 = math.log10(1 + P2x * C12 / (noise * BW2)) * BW2
        elif P2 == a[3] or P2 == a[4]:
            pathloss22 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta22 - aa)))
            pathloss22 = pathloss22 + 20 * math.log10(4 * math.pi * F * d22 / C) + NLOS
            C22 = 10 ** (-pathloss22 / 10)
            rate2 = math.log10(1 + P2x * C22 / (noise * BW2)) * BW2
        else:
            print(' #### warning ####')

        if P3 == 0 or BW3 ==0 :
            rate3 = 0
        elif P3 == a[1] or P3 == a[2]:
            pathloss13 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta13 - aa)))
            pathloss13 = pathloss13 + 20 * math.log10(4 * math.pi * F * d13 / C) + NLOS
            C13 = 10 ** (-pathloss13 / 10)
            rate3 = math.log10(1 + P3x * C13 /  (noise * BW3)) * BW3
        elif P3 == a[3] or P3 == a[4]:
            pathloss23 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta23 - aa)))
            pathloss23 = pathloss23 + 20 * math.log10(4 * math.pi * F * d23 / C) + NLOS
            C23 = 10 ** (-pathloss23 / 10)
            rate3 = math.log10(1 + P3x * C23 / (noise * BW3)) * BW3
        else:
            print(' #### warning ####')

        if P4 == 0 or BW4 ==0 :
            rate4 = 0
        elif P4 == a[1] or P4 == a[2]:
            pathloss14 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta14 - aa)))
            pathloss14 = pathloss14 + 20 * math.log10(4 * math.pi * F * d14 / C) + NLOS
            C14 = 10 ** (-pathloss14 / 10)
            rate4 = math.log10(1 + P4x * C14 / (noise * BW4)) * BW4
        elif P4 == a[3] or P4 == a[4]:
            pathloss24 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta24 - aa)))
            pathloss24 = pathloss24 + 20 * math.log10(4 * math.pi * F * d24 / C) + NLOS
            C24 = 10 ** (-pathloss24 / 10)
            rate4 = math.log10(1 + P4x * C24 / (noise * BW4)) * BW4
        else:
            print(' #### warning ####')

        if P5 == 0 or BW5 ==0 :
            rate5 = 0
        elif P5 == a[2]:
            pathloss15 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta15 - aa)))
            pathloss15 = pathloss15 + 20 * math.log10(4 * math.pi * F * d15 / C) + NLOS
            C15 = 10 ** (-pathloss15 / 10)
            rate5 = math.log10(1 + P5x * C15 /(noise * BW5)) * BW5
        elif  P5 == a[4]:
            pathloss25 = (LOS - NLOS) / (1 + aa * math.exp(-bb * (theta25 - aa)))
            pathloss25 = pathloss25 + 20 * math.log10(4 * math.pi * F * d25 / C) + NLOS
            C25 = 10 ** (-pathloss25 / 10)
            rate5 = math.log10(1 + P5x * C25 /(noise * BW5)) * BW5
        else:
            print(' #### warning ####')

        DC1 = np.clip(rate1* 10**(-6) *0.5, 0, D1Gener)
        DC2 = np.clip(rate2* 10**(-6) *0.5, 0, D2Gener)
        DC3 = np.clip(rate3* 10**(-6) *0.5, 0, D3Gener)
        DC4 = np.clip(rate4* 10**(-6) *0.5, 0, D4Gener)
        DC5 = np.clip(rate5* 10**(-6) *0.5, 0, D5Gener)

        self.reward = (DC1+ DC2+ DC3+ DC4+ DC5) * 0.01

        B1state_ = max(B1state - P1x, 0)
        B2state_ = max(B2state - P2x, 0)
        B3state_ = max(B3state - P3x, 0)
        B4state_ = max(B4state - P4x, 0)
        B5state_ = max(B5state - P5x, 0)

        s_ = np.array([B1state_, B2state_, B3state_, B4state_, B5state_, EH1[h], EH2[h], EH3[h], EH4[h], EH5[h], DG1[h], DG2[h], DG3[h], DG4[h], DG5[h], C11, C12, C13, C14, C15, C21, C22, C23, C24, C25])

        if flag == 0 :
            print(h)
            print('B1state:{:.2f}'.format(B1state_),'B2state:{:.2f}'.format(B2state_),'B3state:{:.2f}'.format(B3state_),
                  'B4state:{:.2f}'.format(B4state_),'B5state:{:.2f}'.format(B5state_))
            print('P1:{:.2f}'.format(P1x),'P2:{:.2f}'.format(P2x),'P3:{:.2f}'.format(P3x),'P4:{:.2f}'.format(P4x),'P5:{:.2f}'.format(P5x))
            print('BW1:{:.2f}'.format(BW1/bandwidth), 'BW2:{:.2f}'.format(BW2/bandwidth), 'BW3:{:.2f}'.format(BW3/bandwidth),'BW4:{:.2f}'.format(BW4/bandwidth), 'BW5:{:.2f}'.format(BW5/bandwidth))
            print('DG1:{:.2f}'.format(D1Gener),'DG2:{:.2f}'.format(D2Gener),'DG3:{:.2f}'.format(D3Gener),'DG4:{:.2f}'.format(D4Gener),'DG5:{:.2f}'.format(D5Gener))
            print('DC1:{:.2f}'.format(rate1* 10**(-6)*0.5), 'DC2:{:.2f}'.format(rate2* 10**(-6)*0.5), 'DC3:{:.2f}'.format(rate3* 10**(-6)*0.5),'DC4:{:.2f}'.format(rate4* 10**(-6)*0.5), 'DC5:{:.2f}'.format(rate5* 10**(-6)*0.5))
            print('SUB1:{:.2f}'.format(rate1* 10**(-6)*0.5-D1Gener), 'SUB2:{:.2f}'.format(rate2* 10**(-6)*0.5-D2Gener),
                  'SUB3:{:.2f}'.format(rate3* 10**(-6)*0.5-D3Gener), 'SUB4:{:.2f}'.format(rate4* 10**(-6)*0.5-D4Gener),'SUB5:{:.2f}'.format(rate5* 10**(-6)*0.5-D5Gener))
            print ('Reward:{:.2f}'.format(self.reward))
        return s_,self.reward, done