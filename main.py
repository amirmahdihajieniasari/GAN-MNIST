import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST  # Training dataset
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

torch.manual_seed(0)  # Set for testing purposes, please do not change!


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    """
    visualizing images: given a matrix of images, number of images, and
    size per image, then prints the images.
    """
    image_un_flat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_un_flat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def get_generator_block(input_dim, output_dim):
    """
    input_dim as dimension of the input vector
    output_dim as dimension of the putput vector
    it returns a neural network layer
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),  # normalization
        nn.ReLU(inplace=True)
    )


class Generator(nn.Module):
    """
    z_dim: the dimension of the noise vector
    im_dim: the dimension of the images, 784 because images of MNIST are 28 * 28 pixels
    hidden_dim: the inner layer
    """

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        # Build the neural network
        self.gen = nn.Sequential(
            get_generator_block(z_dim, hidden_dim),
            get_generator_block(hidden_dim, hidden_dim * 2),
            get_generator_block(hidden_dim * 2, hidden_dim * 4),
            get_generator_block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, im_dim),
            nn.Sigmoid()

        )

    def forward(self, noise):
        """
        returns generated images
        noise: a noise matrix with dimensions
        """
        return self.gen(noise)

    # Needed for grading
    def get_gen(self):
        """
        returns the sequential model
        """
        return self.gen


def get_noise(n_samples, z_dim):
    """
    for creating noise vectors
    n_samples: the number of noise vectors to generate
    z_dim: the dimension of the noise vector
    device: the device type of running
    """
    return torch.randn(n_samples, z_dim, device='cuda')


def get_discriminator_block(input_dim, output_dim):
    """
    for returning a neural network of the discriminator
    input_dim: the dimension of the input vector
    output_dim: the dimension of the output vector
    Returns:
        a discriminator neural network layer, with a linear transformation
          followed by a nn.LeakyReLU activation with negative slope of 0.2
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(0.2, inplace=True)
    )


class Discriminator(nn.Module):
    """
    im_dim: the dimension of the images, 784 for 28 * 28 pixels of MNIST images
    hidden_dim: the inner dimension
    """

    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            get_discriminator_block(im_dim, hidden_dim * 4),
            get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            get_discriminator_block(hidden_dim * 2, hidden_dim),
            # Hint: You want to transform the final output into a single value,
            #       so add one more linear map.
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        """
        Given an image matrix,
        returns a vector representing fake/real
        image: a flattened image matrix with dimension im_dim
        """
        return self.disc(image)

    # Needed for grading
    def get_disc(self):
        """
        Returns:
            the sequential model
        """
        return self.disc


criterion = nn.BCEWithLogitsLoss()  # cost function
n_epochs = 200  # how many times does every single examples of mnist datasets will be updated
z_dim = 64
display_step = 500
batch_size = 128
''' 
batch size = number of examples per batches, the iteration would be 469 cause it took 469              
batches to reach one epoch of training
'''
lr = 0.00001  # as learning rate

# load MNIST dataset as matrix
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True)
gen = Generator(z_dim).to('cuda')
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to('cuda')
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim):
    """
    the loss of the discriminator
    criterion: using for comparing real and fake images for discriminator
    real: a batch of real images(128 images in each batch) given from dataloader
    num_images: the number of images the generator should produce due to batch size,
            which is also the length of the real images
    z_dim: the dimension of the noise vector
    disc_loss: loss value for the current batch
    """
    fake_noise = get_noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())  # stop generator with detach to avoid calling gen class
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    """ comparing fake with grand truth matrix (all zeros with dimension of disc_fake_pred created matrix"""
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    """ comparing real with grand truth matrix (all ones with dimension of disc_fake_pred created matrix"""
    disc_loss = (disc_fake_loss + disc_real_loss) / 2  # main cost function

    return disc_loss


def get_gen_loss(gen, disc, criterion, num_images, z_dim):
    """
    gen: the generator model, which returns an image given z-dimensional noise
    disc: the discriminator model, which returns a single-dimensional prediction of real/fake
    criterion: the loss function, which should be used to compare
            the discriminator's predictions to the ground truth reality of the images
            (e.g. fake = 0, real = 1)
    num_images: the number of images the generator should produce due to batch size,
            which is also the length of the real images
    z_dim: the dimension of the noise vector
    Returns: a loss value for the current batch
    """

    fake_noise = get_noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))

    return gen_loss


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
test_generator = True  # Whether the generator should be tested
gen_loss = False

for epoch in range(n_epochs):

    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):

        # tqdm() for show the progress, get only real(first output of dataloader() with input of MNIST dataset)
        cur_batch_size = len(real)  # 128
        # epoch is 200, data examples is 60000, batch size is 128, so the iteration will be 469
        # Flatten the batch of real images from the dataset
        real = real.view(cur_batch_size, -1).to('cuda')

        # Zero out the gradients before backpropagation
        disc_opt.zero_grad()

        # Calculate discriminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim)

        # Update gradients while keep last forward pass
        disc_loss.backward(retain_graph=True)

        # Update parameters of models with optimizer
        disc_opt.step()

        # Update generator

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim)
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step

        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step

        # Visualization code
        if cur_step % display_step == 0 and cur_step > 0:
            print(
             f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1
