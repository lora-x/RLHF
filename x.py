import numpy as np
l = [[[0.09586896033826484, 0.7029199617569621, 0.9858001857223836, -0.16792258284612566, -1.6151737102765626], [0.10797518967939508, 0.733710869159409, 0.9805913092428065, -0.19606295988145916, -1.73450809790676], [0.12003735997184747, 0.7310406237849935, 0.9743585351419114, -0.22500098887806821, -1.7941041982089487]], [[0.06310342905720598, 0.7032402216675254, 0.986791173818567, -0.16199746687456215, -1.6744402995407024], [0.07400814104920111, 0.6608916358784933, 0.9820455593222305, -0.18864389578114515, -1.6403965502846236], [0.08619843077972451, 0.7388054382135394, 0.9757821478911225, -0.2187445996064531, -1.863434003501556]]]

a = np.asarray(l, dtype=np.float32)

print(a)


for i in l: 
    for j in i:
        print(j)
# ======================================================================

# import gym
# from pprint import pprint


# env = gym.make("Pendulum-v1")
# pprint(vars(env.action_space))

# """
# {'_np_random': None,
#  '_shape': (1,),
#  'bounded_above': array([ True]),
#  'bounded_below': array([ True]),
#  'dtype': dtype('float32'),
#  'high': array([2.], dtype=float32),
#  'low': array([-2.], dtype=float32)}
# """

# ======================================================================

# import torch

# a = torch.randn(2, 3, 4)
# print(a)
# print(a.shape)
# b = a.view (-1, a.shape[-1])
# print(b)
# print(b.shape)
# c = b.view(2, 3, 4)
# print(c==a)

# print(a)
# print(torch.max(a, -1))
# print(torch.max(a, -1).values)


# import torch.nn as nn

# loss = nn.CrossEntropyLoss()
# input = torch.randn(3, 5, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(5)
# print("target: ", target)
# output = loss(input, target)
# output.backward()

# # ======================================================================

# import torch

# log_p1 = torch.tensor([1, 2, 3])
# log_p2 = torch.tensor([-1, 3, 5])
# mu1 = torch.tensor([1, 0, 1])
# mu2 = torch.tensor([0, 1, 0])

# predicted_preference = torch.argmax(torch.stack((log_p1, log_p2), dim = -1), dim = -1)
# actual_preferences = torch.argmax(torch.stack((mu1, mu2), dim = -1), dim = -1)
# print(predicted_preference)
# print(actual_preferences)
# total_rigth = torch.sum(torch.eq(predicted_preference, actual_preferences)).item() 
# print(total_rigth)

# # ======================================================================


# import numpy as np

# sum = np.sum(np.array([-2.3176e+00, -3.9874e+00, -4.9025e+00, -5.6422e+00, -5.7052e+00,
#     -5.6659e+00, -5.1667e+00, -4.6803e+00, -3.8509e+00, -2.9475e+00,
#     -1.5807e-01,  4.7564e-01,  5.0795e-02,  4.3352e-02,  3.7875e-01,
#     5.1088e-01,  3.3633e-02,  9.0134e-01,  4.8327e-02,  6.2335e-01]))

# print (sum)

# # ======================================================================

