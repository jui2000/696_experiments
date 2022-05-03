import matplotlib.pyplot as plt

def loss_visualize(train_loss, name, title, with_clusters = False):
	path = "/mnt/nfs/work1/mccallum/jbshah/696_experiments/alphanum/figures/baseline/"
	if with_clusters == True:
		path = "/mnt/nfs/work1/mccallum/jbshah/696_experiments/alphanum/figures/with_clusters/"
	plt.clf()
	plt.plot(train_loss)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.title(title)
	plt.savefig(path + name + "_train_loss.pdf")

	return None

def acc_visualize(accuracies, name, labels, title, with_clusters = False):
	path = "/mnt/nfs/work1/mccallum/jbshah/696_experiments/alphanum/figures/baseline/"
	if with_clusters == True:
		path = "/mnt/nfs/work1/mccallum/jbshah/696_experiments/alphanum/figures/with_clusters/"
	plt.clf()
	plt.title(title)
	for i in range(len(accuracies)):
		plt.plot(accuracies[i], label=labels[i])
		plt.xlabel("Epochs")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.savefig(path + name + "_train_accuracies.pdf")			
	return None
