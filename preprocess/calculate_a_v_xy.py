from decode import decode_all
import math
import numpy as np
import pandas as pd


path = 'loco_20170215_02.mat'
chan_names, _, finger_pos, spikes, t, _, _ = decode_all(path)


# 计算t整个时间以及整个时间上有多少个window
# unit of t: second
# choose 64ms as window
start_time = t[0]
end_time = t[len(t)-1]
duration = end_time - start_time
num_window = math.floor(duration/(64/1000))
print(duration, num_window)


# 分别计算5627个64ms中，每一个神经元在每一个64毫秒的发放率
# 对每一个神经元，统计在每个64ms上有多少个spike
# 以第一个神经元为例
# “有多少个spike”可以从spike[0][1], spike[0][2], spike[0][3], spike[0][4]中获得(不包含第一列)



# 解码速度，加速度和位移
# 可以看出，每个时间戳间相隔4ms，总共280000个时间戳
# 所以，对于finger_pose和t，只需要取索引为0，16，32···280000的数据在计算每个64ms内的位移等等
new_finger_pose = finger_pos[0:280000:16]
new_t = t[0:280000:16]

# 通过new_finger_pose计算速度，加速度和位移等
# 根据官网，finger_pose是(z, -x, -y)
# 所以现根据(z, -x, -y)获得(x, y)
new_finger_pose = -new_finger_pose[:, 1:3]
new_finger_pose_1 = np.delete(new_finger_pose, 0, axis=0)  # 删除第一行
new_finger_pose_2 = np.delete(new_finger_pose, len(new_finger_pose)-1, axis=0)  # 删除最后一行
xy_dis = new_finger_pose_1 - new_finger_pose_2  # 位移
velocity = xy_dis / 0.064  # 速度
velocity_1 = np.delete(velocity, 0, axis=0)  # 删除第一行
velocity_2 = np.delete(velocity, len(velocity)-1, axis=0)  # 删除最后一行
delta_v = np.insert((velocity_1 - velocity_2), 0, values=velocity[0]*2, axis=0)
acceleration = delta_v / 0.064

pd.DataFrame(data=chan_names).to_csv("chan_names.csv")
pd.DataFrame(data=xy_dis).to_csv("xy_dis.csv")
pd.DataFrame(data=velocity).to_csv("velocity.csv")
pd.DataFrame(data=acceleration).to_csv("acceleration.csv")
pd.DataFrame(data=new_finger_pose).to_csv("new_finger_pose.csv")
pd.DataFrame(data=new_t).to_csv("new_t.csv")


# 还有计算每个64ms内 每个神经元的发放率（排除小于0.5Hz的神经元）
new_spikes = np.empty([len(chan_names), len(new_t)-1])
for n in range(len(chan_names)):
    spike = spikes[n]
    for w in range(len(new_t)-1):
        low = 2795 + w*0.064
        high = low + 0.064
        num_spike = 0
        for m in range(1, 5):
            try:
                for k in spike[m]:
                    if k < low:
                        pass
                    elif low <= k < high:
                        num_spike = num_spike + 1
                    elif k >= high:
                        break
            except TypeError:
                pass
        freq = num_spike/0.064
        new_spikes[n, w] = freq
        print(n, w, freq)
print(new_spikes)

pd.DataFrame(data=new_spikes).to_csv("new_spikes.csv")

