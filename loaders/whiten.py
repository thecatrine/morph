import zipfile

from sklearn.preprocessing import StandardScaler

#from loaders import twitch
import random
import torch
import pickle

def batch_generator(image_generator, batch_size, max_timesteps=150):
    # batch of inputs, batch of outputs

    go = True

    while go:
        batch_steps = []
        batch_inputs = []
        batch_outputs = []

        i = 0
        while i < batch_size:
            try:
                current_image = next(image_generator)
                if current_image.shape != (3, 28 , 28):
                    print("Skipping because of shape: ", current_image.shape)
                    continue
            except StopIteration:
                print("Reached end of archive")
                return
            except Exception as e:
                print("Skipping because of exception")
                print(e)
                print("---")
                continue
            i += 1

            batch_steps.append(0) 
            batch_inputs.append(current_image.unsqueeze(0))
            batch_outputs.append(current_image.unsqueeze(0))

        yield (torch.tensor(batch_steps), torch.cat(batch_inputs), torch.cat(batch_outputs))

class Whitener:
    def __init__(self):
        with open('loaders/constants/scaler.p', 'rb') as f:
            self.scaler = pickle.load(f)

    def a(self, orig, func):
        _, height, width = orig.shape
        image = torch.permute(orig, (1, 2, 0))
        image = image.reshape(height*width, -1)

        image = func(image)
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)

        image = image.reshape(height, width, -1)
        image = torch.permute(image, (2, 0, 1))

        return image
        
    def transform(self, orig_image):
        return self.a(orig_image, self.scaler.transform)
    
    def untransform(self, transformed):
        return self.a(transformed, self.scaler.inverse_transform)

    def differentiable_untransform(self, transformed):
        def differentiable_inverse(tensor):
            X = tensor
            Y = X * torch.tensor(self.scaler.scale_, dtype=tensor.dtype).detach()
            Z = Y + torch.tensor(self.scaler.mean_, dtype=tensor.dtype).detach()
            return Z
        
        return self.a(transformed, differentiable_inverse)
    

# if __name__ == "__main__":
#     z_file = zipfile.ZipFile('loaders/data/twitch_archive.zip', 'r')

#     N_train, training_image_generator, N_test, test_image_generator, N_validate, validation_image_generator = twitch.get_from_zip(z_file)

#     batch_generator = batch_generator(training_image_generator, batch_size=1024)
#     s, inputs, outputs = next(batch_generator)

#     print("input shape: ", inputs.shape)
#     inputs = torch.permute(inputs, dims=(0, 2, 3, 1))
#     print("input shape: ", inputs.shape)
#     inputs = inputs.reshape(1024*28*28, -1)

#     scaler = StandardScaler()
#     scaler.fit(inputs)

#     print("Mean: ", scaler.mean_)

#     import matplotlib.pyplot as plt
#     white = Whitener(28, 28)
#     im = outputs[0]
#     print("image shape: ", im.shape)
#     im2 = white.transform(im)

#     plt.imshow(twitch.tensor_to_image(im2))
#     plt.show()

#     im3 = white.untransform(im2)

#     plt.imshow(twitch.tensor_to_image(im3))
#     plt.show()

#     import pdb; pdb.set_trace()
#     #with open('loaders/constants/scaler.p', 'wb') as f:
#         #pickle.dump(scaler, f)