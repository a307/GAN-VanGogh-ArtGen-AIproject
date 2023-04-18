import re
import matplotlib.pyplot as plt

# read the text file containing D Loss and G Loss data
with open('C:\\Users\\sburk\\PycharmProjects\\GANstats\\gan_stats.txt', 'r') as f:
    data = f.read()

# Extract discriminator and generator loss values
d_loss_values = re.findall(r'\[D loss: ([0-9]+\.[0-9]+)\]', data)
g_loss_values = re.findall(r'\[G loss: ([0-9]+\.[0-9]+)\]', data)

# calculate number of epochs
num_epochs = len(d_loss_values) // 3

if not d_loss_values or not g_loss_values:
    print("No matches found for D Loss and/or G Loss values")
else:
    # convert the loss values from strings to floats
    d_loss_values = [float(val) for val in d_loss_values]
    g_loss_values = [float(val) for val in g_loss_values]

    # create a list of epoch numbers
    epochs = list(range(num_epochs))

    # plot D Loss and G Loss values
    plt.plot(epochs, d_loss_values[:num_epochs*3:3], label='D Loss')
    plt.plot(epochs, g_loss_values[:num_epochs*2:2], label='G Loss')

    # set x and y labels
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # set the title of the plot
    plt.title('GAN Loss Graph')

    # show the legend
    plt.legend()

    # display the plot
    plt.show()
