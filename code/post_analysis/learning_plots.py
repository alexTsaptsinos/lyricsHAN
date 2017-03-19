import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
# large dataset
large_line_trainloss = np.array([2.2036, 2.20339, 1.9626, 1.9290, 1.9067, 1.8873, 1.8684, 1.8522, 1.8339, 1.8194])

large_line_devloss = np.array([1.9820, 1.93728, 1.9307, 1.9093, 1.9076, 1.9123, 1.9134, 1.9079, 1.9006, 1.9063])

large_section_trainloss = np.array([2.288505210334031, 2.1069748785247278, 2.027044660689208, 1.984929043272893, 1.9506481677303869, 1.9226140224466273, 1.8890818868782906, 1.8597170263253937, 1.8298372551912163, 1.800919015930367])

large_section_devloss = np.array([2.0690575578671795, 1.991171054079846, 1.9823722338914027, 1.9657367542601598, 1.9807712941336808, 1.9656527486885143, 1.9842414859788895, 1.9776791591270297, 2.001127179509687, 1.9983207912776142])

# top 20

top20_line_trainloss = np.array([1.7720450764980173, 1.6198353595649757, 1.5642071342140444, 1.5241577503598114, 1.4933839005671685, 1.4632599995230178, 1.4379154573171335, 1.4152279803139336, 1.392806219473147, 1.3723069271683113])

top20_line_devloss = np.array([1.6074728673981125, 1.5560486996428389, 1.523711210691138, 1.5508364125404268, 1.510617610385539, 1.5113721124750208, 1.4952309763510614, 1.5066696735880782, 1.5055744383834877, 1.505014346259143])

top20_section_trainloss = np.array([1.8443745517244632, 1.671580620022882, 1.6065420995726534, 1.5588319284208636, 1.5170241428570654, 1.4740214555861535, 1.4348477637983847, 1.394961651750393, 1.3571605053676046, 1.3221086378116946])

top20_section_devloss = np.array([1.6513520491731815, 1.6257909643611441, 1.5817490834109493, 1.5618905628814717, 1.5647922441390587, 1.5562386188354098, 1.5703230590436599, 1.5854498484160784, 1.5861032329797036, 1.6046116952080989])


# now plot
#fig, ax = plt.subplots()
plt.figure(figsize=(30,15))
plt.subplot(2,3,1)
axis_fontsize = 20
title_fontsize = 24
l1 = plt.plot(large_line_trainloss, color='orange')
l2 = plt.plot(large_line_devloss, color='blue')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title("117 Genre over Lines", fontsize=24)

plt.subplot(2,3,2)
plt.plot(large_section_trainloss, color='orange')
plt.plot(large_section_devloss, color='blue')
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.title("117 Genre over Sections",fontsize=24)

plt.subplot(2,3,4)
plt.plot(top20_line_trainloss, color='orange')
plt.plot(top20_line_devloss, color='blue')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title("20 Genre over Lines", fontsize=24)

plt.subplot(2,3,5)
plt.plot(top20_section_trainloss, color='orange')
plt.plot(top20_section_devloss, color='blue')
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.title("20 Genre over Sections", fontsize=24)

orange_patch = mlines.Line2D([], [], color='orange', label='Test Accuracy')
blue_patch = mlines.Line2D([], [], color='blue', label='Dev Accuracy')
labels = ['Test Accuracy', 'Dev Accuracy']
plt.figlegend((orange_patch, blue_patch), labels, loc = (0.66, 0.47), fontsize=24)
#plt.show()
plt.savefig('learning_plots.pdf')