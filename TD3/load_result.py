import pickle
import collections

Data = collections.namedtuple("Data", ["steps", "reward"])
# data = pickle.load(open("TD3_Hopper-v2_2.pkl", 'rb'))
# print(data[-10:])
# len(data)
data = pickle.load(open("TD3_HalfCheetah-v2_0.pkl", 'rb'))
print(data[-10:])
len(data)

data = pickle.load(open("TD3_Hopper-v2_0.pkl", 'rb'))
print(data[-10:])
len(data)

data = pickle.load(open("/home/ganche/Downloads/project/TD3/td3_Ant-v2_seed3.pickle", 'rb'))
print(data[-10:])
len(data)

data = pickle.load(open("TD3_Walker2d-v2_0.pkl", 'rb'))
print(data[-10:])
len(data)
data = pickle.load(open("TD3_Reacher-v2_0.pkl", 'rb'))
print(data[-10:])
len(data)

data = pickle.load(open("TD3_InvertedDoublePendulum-v2_0.pkl", 'rb'))
print(data[-10:])
