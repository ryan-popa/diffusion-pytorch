import streamlit as st
import torch
from torchvision import datasets, transforms
from PIL import Image
import io
from model import VAE 
import random

def display_img(img):
    max_width = 450

def main():
    # Load VAE model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu")
    model = VAE().to(device)

    option = st.radio(
        'Choose a model weight to load:',
        ('MNIST', 'FASHION_MNIST'))

    if option == 'MNIST':
        model.load_state_dict(torch.load('vae_model.pth'))
    elif option == 'FASHION_MNIST':
        model.load_state_dict(torch.load('vae_model_fashionmnist.pth'))
    else:
        return

    model.eval()

    # MNIST data loading
    mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    # FashionMNIST data loading
    fashion_mnist_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
    fashion_loader = torch.utils.data.DataLoader(fashion_mnist_test, batch_size=1, shuffle=True)

    # Streamlit UI
    st.title('Variational Autoencoder Inference')

    img_tensor = None
    c1, c2 = st.columns(2)
    with c1:
        if st.button('Sample Random MNIST Image'):
            random_index = random.randint(0, len(fashion_mnist_test)-1)
            img_tensor, _ = mnist_dataset[random_index]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            image = transforms.ToPILImage()(img_tensor.squeeze())

    with c2:
        if st.button('Sample Random FashionMNIST Image'):
            random_index = random.randint(0, len(fashion_mnist_test)-1)
            img_tensor, _ = fashion_mnist_test[random_index]
            img_tensor = img_tensor.unsqueeze(0).to(device)
            image = transforms.ToPILImage()(img_tensor.squeeze())
        
    if img_tensor is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.image(image, caption='Sampled image:', use_column_width=True)
        with torch.no_grad():
            reconstruction, mu, logvar = model(img_tensor)
            st.write("mean: ", mu)
            st.write("logvar: ", logvar)
        reconstructed_img = transforms.ToPILImage()(reconstruction.cpu().squeeze(0).view(28, 28))
        with c2:
            st.image(reconstructed_img, caption='Reconstructed Image.', use_column_width=True)

main()
