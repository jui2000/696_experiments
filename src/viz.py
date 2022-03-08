import matplotlib.pyplot as plt

def loss_visualize(train_loss):
	plt.plot(train_loss)
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.savefig("figures/train_loss.pdf")

	return None

def acc_visualize(accuracies, labels):
	for i in range(len(accuracies)):
		plt.plot(accuracies[i], label=labels[i])
		plt.xlabel("Iterations")
		plt.ylabel("Accuracy")
		plt.legend(loc="upper right")
		plt.savefig("figures/train_accuracies.pdf")			

	return None
