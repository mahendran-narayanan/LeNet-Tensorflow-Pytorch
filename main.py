import argparse
import tensorflow as tf
import torch


def conv_layer(*args):
	if len(args)==4:
		return tf.keras.layers.Conv2D(args[0],args[1],strides=args[2],activation=args[3])
	else:
		return torch.nn.Conv2d(args[0], args[1], kernel_size=args[2])

class LeNet_tf(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_layer(6,5,1,'relu')
		self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
		self.conv2 = conv_layer(16,5,1,'relu')
		self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))
		self.flat = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(120,activation='relu')
		self.dense2 = tf.keras.layers.Dense(84,activation='relu')
		self.classifier = tf.keras.layers.Dense(10,activation='sigmoid')

	def call(self,x):
		x = self.maxpool1(self.conv1(x))
		x = self.maxpool2(self.conv2(x))
		x = self.flat(x)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.classifier(x)
		return x

class LeNet_torch(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = conv_layer(1,6,5)
		self.relu1 = torch.nn.ReLU()
		self.maxpool1 = torch.nn.MaxPool2d(2,2)
		self.conv2 = conv_layer(6,16,5)
		self.relu2 = torch.nn.ReLU()
		self.maxpool2 = torch.nn.MaxPool2d(2,2)
		self.flatten = torch.nn.Flatten()
		self.dense1 = torch.nn.Linear(16*16,120) # 4*4*16
		self.dense2 = torch.nn.Linear(120,84)
		self.classifier = torch.nn.Linear(84,10)
		
	def forward(self, x):
		x = self.relu1(self.conv1(x))
		x = self.maxpool1(x)
		x = self.relu2(self.conv2(x))
		x = self.maxpool2(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.classifier(x)
		return x

def model_torch():
	return LeNet_torch()

def model_tf():
	return LeNet_tf()

def main(args):
	if args.model=='tf':
		print('Model will be created in Tensorflow')
		model = model_tf()
		model.build(input_shape=(None,28,28,1))
		model.summary()
	else:
		print('Model will be created in Pytorch')
		model = model_torch()
		print(model)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create LeNet model in Tensorflow or Pytorch')
	parser.add_argument('--model',
	                    default='tf',
	                    choices=['tf', 'torch'],
	                    help='Model will be created on Tensorflow, Pytorch (default: %(default)s)')
	args = parser.parse_args()
	main(args)